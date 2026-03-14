from fastapi import FastAPI, Body
import uvicorn
from rag_pipeline import stage1, stage2, stage3
app = FastAPI(title="Shopping Assistant API")

@app.post("/stage1")
def stage1_api(input =Body(...)):
    result = stage1(input)
    return result

@app.post("/stage2")
def stage2_api(stage1_res=Body(...)):
    candidates = stage2(stage1_res)
    return candidates

@app.post("/stage3")
def stage3_api(stage2_res=Body(...)):
    stage3_res = stage3(stage2_res)
    return stage3_res

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)