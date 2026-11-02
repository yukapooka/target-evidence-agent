"""Open Targets wrappers: disease lookup, target-disease associations, known drugs and target aliases.

Objective:
    Turn Open Targets Platform GraphQL answers into small, typed Pydantic objects
    that the ranking and critic agents can use as tools.

Functions:
    search_disease()          disease name -> top matching disease IDs (the agent picks one)
    get_associated_targets()  disease ID -> linked genes with overall and per-evidence-type scores
    get_known_drugs()         target ID + disease ID -> drugs tested for that disease, with the stage reached
    get_target_info()         target ID -> official symbol, name and aliases (for checking abstracts)

Rationale:
    - Disease IDs are always looked up, never hardcoded, so any condition can be the input.
    - Every function follows one pattern: query string -> graphql() -> Pydantic model.
      Queries use GraphQL variables rather than pasting values into the string.
    - Only the fields the agents need are requested, which keeps answers small and the models simple.
    - Open Targets returns null (not an error) for unknown IDs, so wrappers raise a clear ValueError.
"""

from pydantic import BaseModel

from agent.tools.http import graphql


# --- Step 2: disease search ---
# Turns a disease name into Open Targets IDs. Returns several hits so the agent can pick the right one.
# In:  search_disease("atopic dermatitis")
# Out: [DiseaseHit(id="MONDO_0004980", name="atopic eczema", description="A common chronic ..."),
#       DiseaseHit(id="MONDO_0002406", name="dermatitis", ...), ...]   (5 hits)

SEARCH_DISEASE_QUERY = """
query SearchDisease($query: String!, $size: Int!) {
  search(queryString: $query, entityNames: ["disease"], page: {index: 0, size: $size}) {
    hits { id name description }
  }
}
"""


class DiseaseHit(BaseModel):
    id: str
    name: str
    description: str | None = None


def search_disease(query: str, size: int = 5) -> list[DiseaseHit]:
    """Search Open Targets for diseases matching a name; returns the top hits for the caller to choose from."""
    data = graphql(SEARCH_DISEASE_QUERY, {"query": query, "size": size})
    return [DiseaseHit(**hit) for hit in data["search"]["hits"]]


# --- Step 3: associated targets ---
# Genes linked to a disease, with an overall score and a score per evidence type
# (genetic_association, literature, clinical, ...). This is the agent's candidate list.
# In:  get_associated_targets("MONDO_0004980")
# Out: [TargetAssociation(ensembl_id="ENSG00000143631", symbol="FLG", name="filaggrin", overall_score=0.84,
#                         datatype_scores={"genetic_association": 0.95, "literature": 0.98, ...}),
#       TargetAssociation(symbol="IL13", overall_score=0.75, datatype_scores={"clinical": 0.98, ...}), ...]  (25 rows)
# In:  get_associated_targets("NOT_A_DISEASE")  -> raises ValueError

ASSOCIATED_TARGETS_QUERY = """
query AssociatedTargets($diseaseId: String!, $size: Int!) {
  disease(efoId: $diseaseId) {
    associatedTargets(page: {index: 0, size: $size}) {
      rows {
        score
        target { id approvedSymbol approvedName }
        datatypeScores { id score }
      }
    }
  }
}
"""


class TargetAssociation(BaseModel):
    ensembl_id: str
    symbol: str
    name: str
    overall_score: float
    datatype_scores: dict[str, float]


def get_associated_targets(disease_id: str, size: int = 25) -> list[TargetAssociation]:
    """Targets linked to a disease, highest overall association score first, with scores by evidence type."""
    data = graphql(ASSOCIATED_TARGETS_QUERY, {"diseaseId": disease_id, "size": size})
    # An unknown ID comes back as null rather than an error, so raise a clear one here.
    if data["disease"] is None:
        raise ValueError(f"Unknown disease ID: {disease_id}")
    return [
        TargetAssociation(
            ensembl_id=row["target"]["id"],
            symbol=row["target"]["approvedSymbol"],
            name=row["target"]["approvedName"],
            overall_score=row["score"],
            # The API sends [{"id": "clinical", "score": 0.98}, ...]; a dict is easier to look up.
            datatype_scores={d["id"]: d["score"] for d in row["datatypeScores"]},
        )
        for row in data["disease"]["associatedTargets"]["rows"]
    ]


# --- Step 4: known drugs (clinical status) ---
# Drugs acting on one target, kept only if they were tested for the given disease.
# Clinical status comes from drugAndClinicalCandidates (Open Targets removed the old knownDrugs field).
# The row's maxClinicalStage covers ALL diseases, so the disease-specific stage is taken from the
# trial reports for this disease (e.g. cendakimab: PHASE_3 overall, PHASE_2 for atopic eczema).
# In:  get_known_drugs("ENSG00000169194", "MONDO_0004980")   (IL13, atopic eczema)
# Out: [KnownDrug(drug_id="CHEMBL1743035", drug_name="LEBRIKIZUMAB", drug_type="Antibody",
#                 stage_for_disease="APPROVAL", max_stage_any_disease="PHASE_4",
#                 mechanisms=["Interleukin-13 inhibitor"], drug_targets=["IL13"]),
#       KnownDrug(drug_name="TRALOKINUMAB", ...), KnownDrug(drug_name="CENDAKIMAB", stage_for_disease="PHASE_2", ...)]
#       (highest stage first; IL13 drugs never tested in atopic eczema are left out)

KNOWN_DRUGS_QUERY = """
query KnownDrugs($ensemblId: String!) {
  target(ensemblId: $ensemblId) {
    drugAndClinicalCandidates {
      rows {
        maxClinicalStage
        drug { id name drugType mechanismsOfAction { rows { mechanismOfAction targets { approvedSymbol } } } }
        diseases { disease { id } }
        clinicalReports { clinicalStage diseases { disease { id } } }
      }
    }
  }
}
"""

# Lowest to highest; used to pick the furthest stage a drug reached for the disease.
STAGE_ORDER = ["UNKNOWN", "EARLY_PHASE_1", "PHASE_1", "PHASE_1_2", "PHASE_2", "PHASE_2_3", "PHASE_3", "PHASE_4", "APPROVAL"]


class KnownDrug(BaseModel):
    drug_id: str
    drug_name: str
    drug_type: str | None = None
    stage_for_disease: str
    max_stage_any_disease: str
    mechanisms: list[str]
    drug_targets: list[str] = []     # symbols of every target the drug acts on (all mechanisms)


def _stage_rank(stage: str) -> int:
    return STAGE_ORDER.index(stage) if stage in STAGE_ORDER else 0


def _mentions_disease(diseases: list[dict] | None, disease_id: str) -> bool:
    # disease can be null when Open Targets couldn't map the trial's condition to an ID.
    return any(d.get("disease") and d["disease"]["id"] == disease_id for d in diseases or [])


def get_known_drugs(ensembl_id: str, disease_id: str) -> list[KnownDrug]:
    """Drugs acting on a target that were tested for the disease, with the furthest stage reached for that disease."""
    data = graphql(KNOWN_DRUGS_QUERY, {"ensemblId": ensembl_id})
    if data["target"] is None:
        raise ValueError(f"Unknown target ID: {ensembl_id}")
    drugs = []
    for row in data["target"]["drugAndClinicalCandidates"]["rows"]:
        if row["drug"] is None:
            continue
        stages = [r["clinicalStage"] for r in row["clinicalReports"] if _mentions_disease(r["diseases"], disease_id)]
        if not stages and not _mentions_disease(row["diseases"], disease_id):
            continue
        drug = row["drug"]
        drugs.append(KnownDrug(
            drug_id=drug["id"],
            drug_name=drug["name"],
            drug_type=drug["drugType"],
            stage_for_disease=max(stages, key=_stage_rank) if stages else "UNKNOWN",
            max_stage_any_disease=row["maxClinicalStage"],
            mechanisms=sorted({m["mechanismOfAction"] for m in (drug["mechanismsOfAction"] or {}).get("rows", [])}),
            drug_targets=sorted({t["approvedSymbol"] for m in (drug["mechanismsOfAction"] or {}).get("rows", [])
                                 for t in m.get("targets") or []}),
        ))
    return sorted(drugs, key=lambda d: _stage_rank(d.stage_for_disease), reverse=True)


# --- Step 5: target info and aliases ---
# A gene's official symbol and name plus its other names, deduplicated.
# Used by the critic to check whether a cited abstract really mentions the target (papers often use aliases).
# In:  get_target_info("ENSG00000077238")   (IL4R)
# Out: TargetInfo(ensembl_id="ENSG00000077238", symbol="IL4R", name="interleukin 4 receptor",
#                 synonyms=["IL-4 receptor subunit alpha", "IL-4R-alpha", "IL-4RA", "CD124", ...])   (17 for IL4R)

TARGET_INFO_QUERY = """
query TargetInfo($ensemblId: String!) {
  target(ensemblId: $ensemblId) {
    id
    approvedSymbol
    approvedName
    symbolSynonyms { label }
    nameSynonyms { label }
  }
}
"""


class TargetInfo(BaseModel):
    ensembl_id: str
    symbol: str
    name: str
    synonyms: list[str]


def get_target_info(ensembl_id: str) -> TargetInfo:
    """Official symbol and name of a target, plus its symbol and name synonyms."""
    data = graphql(TARGET_INFO_QUERY, {"ensemblId": ensembl_id})
    target = data["target"]
    if target is None:
        raise ValueError(f"Unknown target ID: {ensembl_id}")
    labels = [s["label"] for s in (target["symbolSynonyms"] or []) + (target["nameSynonyms"] or [])]
    # dict.fromkeys keeps the first occurrence of each label, in order, and drops the official symbol.
    synonyms = [s for s in dict.fromkeys(labels) if s != target["approvedSymbol"]]
    return TargetInfo(
        ensembl_id=target["id"],
        symbol=target["approvedSymbol"],
        name=target["approvedName"],
        synonyms=synonyms,
    )

