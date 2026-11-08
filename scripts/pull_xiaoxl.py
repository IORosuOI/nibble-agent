import subprocess, os, sys, pyarrow.parquet as pq, pandas as pd
base="https://huggingface.co/datasets/xiaoxl/recipe_data/resolve/main/data/"
shards=[f"test-{i:05d}-of-00004.parquet" for i in range(4)]+[f"train-{i:05d}-of-00016.parquet" for i in range(16)]
cols=["title","url","category","description","ingredients","directions","instructions_list","servings","yields","calories","carbohydrates_g","fat_g","protein_g","dietary_fiber_g","sugars_g","saturated_fat_g","sodium_mg","cholesterol_mg","prep_time","cook_time","total_time"]
raw="xiaoxl_raw"; out="xiaoxl_nutrition"; os.makedirs(out,exist_ok=True)
tot_rows=tot_keep=0
for s in shards:
    f=os.path.join(raw,s)
    if not os.path.exists(f):
        r=subprocess.run(["curl","-sfL","-o",f,base+s]); 
        if r.returncode: print("FAIL",s,flush=True); continue
    d=pq.read_table(f,columns=cols).to_pandas()
    for c in ["servings","calories","carbohydrates_g","fat_g","protein_g"]: d[c]=pd.to_numeric(d[c],errors="coerce")
    k=d.dropna(subset=["servings","calories","carbohydrates_g","fat_g","protein_g"])
    k=k[k.servings>0]
    k.assign(shard=s).to_csv(os.path.join(out,s.replace(".parquet",".csv")),index=False,encoding="utf-8")
    tot_rows+=len(d); tot_keep+=len(k); os.remove(f)
    print(s,len(d),len(k),"| total",tot_rows,tot_keep,flush=True)
print("DONE",tot_rows,tot_keep,flush=True)
