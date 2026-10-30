"""Skin expression wrapper: how strongly a gene is expressed in normal human skin (the omics input).

Objective:
    Give the agents a tissue-level expression record for the "Expression in skin tissue" rubric criterion.

Functions:
    get_skin_expression()  target ID -> TPM in the two GTEx skin tissues, plus where skin ranks among all GTEx tissues

Rationale:
    - Source is Open Targets baselineExpression (GTEx bulk RNA-seq), so it reuses graphql() and the cache;
      no second API is needed.
    - GTEx has two skin tissues: "Skin - Sun Exposed (Lower leg)" and "Skin - Not Sun Exposed (Suprapubic)".
    - skin_rank gives context: 27 TPM means little alone, but "rank 28 of 51 tissues" shows it isn't skin-specific.
    - All rows are fetched in ONE page (size 3000, the API maximum): paging with smaller sizes returned
      duplicated rows and missed the skin rows.
    - Limitation: this is healthy skin, not eczema lesions or scalp hair follicles.
"""

from pydantic import BaseModel

from agent.tools.http import graphql

# --- Step 6: skin expression ---
# GTEx skin TPM for one target, and where skin ranks among all 51 GTEx tissues (other data sources are ignored).
# In:  get_skin_expression("ENSG00000077238")   (IL4R)
# Out: SkinExpression(ensembl_id="ENSG00000077238", symbol="IL4R",
#                     skin_tissues=[TissueTPM(tissue="Skin - Sun Exposed (Lower leg)", uberon_id="UBERON_0004264", median_tpm=27.316),
#                                   TissueTPM(tissue="Skin - Not Sun Exposed (Suprapubic)", uberon_id="UBERON_0036149", median_tpm=25.5316)],
#                     skin_rank=28, gtex_tissue_count=51)

SKIN_EXPRESSION_QUERY = """
query SkinExpression($ensemblId: String!) {
  target(ensemblId: $ensemblId) {
    id
    approvedSymbol
    baselineExpression(page: {index: 0, size: 3000}) {
      rows { datasourceId tissueBiosampleFromSource tissueBiosample { biosampleId } median unit }
    }
  }
}
"""

class TissueTPM(BaseModel):
    tissue: str
    uberon_id: str
    median_tpm: float

class SkinExpression(BaseModel):
    ensembl_id: str
    symbol: str
    skin_tissues: list[TissueTPM]
    skin_rank: int
    gtex_tissue_count: int

def get_skin_expression(ensembl_id: str) -> SkinExpression:
    """GTEx skin TPM for a target, plus where skin ranks among all GTEx tissues (1 = highest)."""
    data = graphql(SKIN_EXPRESSION_QUERY, {"ensemblId": ensembl_id})
    if data["target"] is None:
        raise ValueError(f"Unknown target: {ensembl_id}")
    rows = data["target"]["baselineExpression"]["rows"]
    # rows is a list of dicts (one per tissue), so filter with list comprehensions.
    gtex_rows = [row for row in rows if row["datasourceId"] == "gtex"]
    skin_rows = [row for row in gtex_rows if "Skin" in row["tissueBiosampleFromSource"]]
    # Rank tissues by expression (1 = highest); skin_rank is where the first skin tissue lands. 0 = no skin data.
    sorted_rows = sorted(gtex_rows, key=lambda row: row["median"], reverse=True)
    skin_rank = next((i for i, row in enumerate(sorted_rows, start=1) if "Skin" in row["tissueBiosampleFromSource"]), 0)
    return SkinExpression(
        ensembl_id=data["target"]["id"],
        symbol=data["target"]["approvedSymbol"],
        skin_tissues=[
            TissueTPM(
                tissue=row["tissueBiosampleFromSource"],
                uberon_id=row["tissueBiosample"]["biosampleId"],
                median_tpm=row["median"],
            )
            for row in skin_rows
        ],
        skin_rank=skin_rank,
        gtex_tissue_count=len(gtex_rows),
    )
