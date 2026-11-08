import os, re, pandas as pd, numpy as np
from pathlib import Path
D = str(Path(__file__).resolve().parent.parent / "usda_sr" / "FoodData_Central_sr_legacy_food_csv_2018-04") + os.sep
FRAC = {"\u00bc":.25,"\u00bd":.5,"\u00be":.75,"\u2153":1/3,"\u2154":2/3,"\u215b":.125,"\u215c":.375,"\u215d":.625,"\u215e":.875,"\u2155":.2}
STOP = set("clove cloves extra virgin skinless halves half sprigs sprig bunch head heads leaves leaf fresh large small medium chopped diced minced sliced grated shredded crushed finely coarsely thinly peeled seeded halved cubed softened melted divided drained rinsed packed optional to taste or as needed for garnish plus more lightly beaten room temperature whole extra boneless skinless cut into inch pieces pieces thick thin hot cold warm boiling lukewarm chilled stick sticks about and of the a an taste package packages".split())
MASS = {"ounce":28.3495,"ounces":28.3495,"oz":28.3495,"pound":453.592,"pounds":453.592,"lb":453.592,"lbs":453.592,"gram":1,"grams":1,"g":1,"kilogram":1000,"kilograms":1000}
VOL = {"cup":"cup","cups":"cup","tablespoon":"tbsp","tablespoons":"tbsp","tbsp":"tbsp","teaspoon":"tsp","teaspoons":"tsp","tsp":"tsp","pinch":"pinch","pinches":"pinch","dash":"pinch","quart":"quart","quarts":"quart","pint":"pint","pints":"pint","fluid":"floz"}
ML = {"cup":236.6,"tbsp":14.79,"tsp":4.93,"pinch":0.3,"quart":946,"pint":473,"floz":29.57}
ALIAS = {"flour":"wheat flour white all-purpose enriched bleached","all-purpose flour":"wheat flour white all-purpose enriched bleached","sugar":"sugars granulated","white sugar":"sugars granulated","brown sugar":"sugars brown","butter":"butter salted","egg":"egg whole raw fresh","eggs":"egg whole raw fresh","milk":"milk whole 3.25% milkfat","water":"water","salt":"salt table","baking powder":"leavening agents baking powder double-acting","baking soda":"leavening agents baking soda","vanilla extract":"vanilla extract","olive oil":"oil olive salad or cooking","vegetable oil":"oil vegetable soybean","oil":"oil vegetable soybean","onion":"onions raw","garlic":"garlic raw","cream cheese":"cream cheese","mayonnaise":"salad dressing mayonnaise regular","sour cream":"sour cream cultured","black pepper":"spices pepper black","pepper":"spices pepper black","cinnamon":"spices cinnamon ground","honey":"honey","rice":"rice white long-grain regular raw enriched","chicken broth":"soup chicken broth ready-to-serve","parmesan cheese":"cheese parmesan grated","cheddar cheese":"cheese cheddar","mozzarella cheese":"cheese mozzarella whole milk","chicken breast":"chicken broilers or fryers breast meat only raw","ground beef":"beef ground 85% lean meat 15% fat raw","tomato sauce":"tomato products canned sauce","tomatoes":"tomatoes red ripe raw","potatoes":"potatoes flesh and skin raw","carrots":"carrots raw","cornstarch":"cornstarch","soy sauce":"soy sauce made from soy and wheat shoyu","lemon juice":"lemon juice raw","bread crumbs":"bread crumbs dry grated plain","walnuts":"nuts walnuts english","pecans":"nuts pecans","oats":"cereals oats regular and quick not fortified dry","cocoa":"cocoa dry powder unsweetened","white vinegar":"vinegar distilled","vinegar":"vinegar distilled","green onion":"onions spring or scallions includes tops and bulb raw","green onions":"onions spring or scallions includes tops and bulb raw","garlic salt":"salt table","chicken breast halves":"chicken broilers or fryers breast meat only raw","chicken breast":"chicken broilers or fryers breast meat only raw","lemon":"lemons raw without peel","lime juice":"lime juice raw","cheese":"cheese cheddar","bacon":"pork cured bacon cooked pan-fried","bananas":"bananas raw","banana":"bananas raw","apples":"apples raw with skin","apple":"apples raw with skin","oranges":"oranges raw all commercial varieties","orange juice":"orange juice raw"}

def load():
    f = pd.read_csv(D+"food.csv", usecols=["fdc_id","description"]).astype({"fdc_id":int})
    n = pd.read_csv(D+"food_nutrient.csv", usecols=["fdc_id","nutrient_id","amount"])
    n = n[n.nutrient_id.isin([1008,1003,1004,1005])].pivot_table(index="fdc_id",columns="nutrient_id",values="amount",aggfunc="first")
    n.columns = ["protein","fat","carb","kcal"][:0] or [{1003:"protein",1004:"fat",1005:"carb",1008:"kcal"}[c] for c in n.columns]
    f = f.merge(n, left_on="fdc_id", right_index=True).dropna(subset=["kcal"])
    f["toks"] = f.description.map(lambda s: set(re.findall(r"[a-z0-9%]+", s.lower())))
    mu = pd.read_csv(D+"measure_unit.csv").set_index("id").name.to_dict()
    p = pd.read_csv(D+"food_portion.csv", usecols=["fdc_id","amount","measure_unit_id","modifier","gram_weight"])
    p["unit"] = p.measure_unit_id.map(mu).fillna("")
    return f.reset_index(drop=True), p

def _num(tok):
    tok = tok.strip()
    for k,v in FRAC.items(): tok = tok.replace(k, f" {v} ")
    tot = 0.0; ok = False
    for part in tok.replace("\u2009"," ").split():
        try:
            if "/" in part: a,b = part.split("/"); tot += float(a)/float(b)
            else: tot += float(part)
            ok = True
        except Exception: pass
    return tot if ok else None

def parse(line):
    """-> (qty, unit, name, grams_hint). qty/unit may be None."""
    s = line.lower().replace("\ufffd"," ")
    s = re.sub(r"\s+"," ", s).strip()
    m = re.match(r"^((?:[\d\u00bc-\u00be\u2150-\u215e\u2009\s/.]|\s)+)\s*(.*)$", s)
    qty, rest = (1.0, s)
    if m and _num(m.group(1)) is not None: qty, rest = _num(m.group(1)), m.group(2)
    elif not m or _num(m.group(1)) is None: qty = None if not re.match(r"^\d", s) else 1.0
    # "(10 ounce)" package size
    pk = re.match(r"^\((\d+(?:\.\d+)?)\s*(ounce|ounces|oz|pound|pounds|lb|gram|grams|g)\b[^)]*\)\s*(.*)$", rest)
    if pk: return (qty or 1)*float(pk.group(1))*MASS[pk.group(2)], "g", pk.group(3), True
    rest = re.sub(r"\([^)]*\)", " ", rest)
    w = rest.split()
    unit = None
    if w and w[0] in MASS: unit = w[0]; w = w[1:]
    elif w and w[0] in VOL: unit = w[0]; w = w[1:]; 
    if w and w[0] == "of": w = w[1:]
    name = " ".join(w).split(",")[0]
    return qty, unit, name.strip(), False

def best_food(name, F, cache={}):
    if name in cache: return cache[name]
    base = re.sub(r"[^a-z ]"," ", name).split()
    key = " ".join(w for w in base if w not in STOP)
    q = ALIAS.get(key) or ALIAS.get(re.sub(r"s$","",key)) or key
    qt = set(re.findall(r"[a-z0-9%]+", q.lower()))
    if not qt: cache[name]=None; return None
    sc = F.toks.map(lambda t: len(t & qt)/ (len(qt) + 0.15*len(t - qt)))
    i = sc.idxmax(); res = (i, sc[i]) if sc[i] >= .6 else None
    cache[name] = res; return res

def grams(qty, unit, name, hint, fid, P, F):
    if hint: return qty
    if qty is None: qty = 1.0
    if unit in MASS: return qty*MASS[unit]
    pf = P[P.fdc_id==fid]
    if unit in VOL:
        u = VOL[unit]
        key = {"cup":"cup","tbsp":"tablespoon","tsp":"teaspoon"}.get(u)
        if key:
            r = pf[pf.unit.eq(key)|pf.modifier.fillna("").str.contains(key)]
            if len(r): x = r.iloc[0]; return qty*x.gram_weight/max(x.amount,1e-9)
        return qty*ML[u]*(1.0)  # assume ~1 g/ml
    # counted item: first portion
    if len(pf): x = pf.iloc[0]; return qty*x.gram_weight/max(x.amount,1e-9)
    return None

def recipe_macros(lines, F, P):
    tot = dict(kcal=0.,protein=0.,fat=0.,carb=0.); ok = 0; miss = []
    lines=[l for l in lines if not re.search(r"to taste|for garnish|as needed|cooking spray", l.lower())]
    for ln in lines:
        qty, unit, name, hint = parse(ln)
        if not name or name in ("salt","water","cooking spray") and False: pass
        bf = best_food(name, F)
        if not bf: miss.append(ln); continue
        fid = F.fdc_id.iloc[bf[0]]
        g = grams(qty, unit, name, hint, fid, P, F)
        if g is None or not np.isfinite(g) or g > 20000: miss.append(ln); continue
        row = F.iloc[bf[0]]
        for k in tot: tot[k] += row[k]*g/100.0
        ok += 1
    return tot, ok/max(len(lines),1), miss
