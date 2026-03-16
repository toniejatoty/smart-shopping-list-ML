from fastapi import FastAPI, Body
import uvicorn
from rag_pipeline import stage1, stage3
app = FastAPI(title="Shopping Assistant API")

@app.post("/stage1")
def stage1_api(input =Body(...)):
    """Etap 1 + 2: Analiza historii, intencji i kandydatów produktów."""
    result = stage1(input)
    return result

@app.post("/stage3")
def stage3_api(stage2_candidates=Body(...), history_list = Body(...)):
    stage3_res = stage3(stage2_candidates, history_list)
    return stage3_res

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)