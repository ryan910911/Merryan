"""manifest + 上傳對照 + queue/*.json → ArtifactData batch 檔（photos 與 listings）"""
import json, os, sys, time, glob
man = json.load(open(sys.argv[1])); up = dict(l.split() for l in open(sys.argv[2])); qdir = sys.argv[3]
os.makedirs('docs', exist_ok=True); base = int(time.time() * 1000); w = []
for i, x in enumerate(man):
    d = {"asset": up[x['id']], "w": x['w'], "h": x['h'], "thumb": x['thumb'], "group": x['group'], "n": x['n'],
         "side": None, "psa": None, "edges": None, "corners": None, "surface": None, "marks": [], "done": False,
         "createdAt": base - i, "updatedAt": base}
    p = os.path.abspath(f"docs/{x['id']}.json"); json.dump(d, open(p, 'w'))
    w.append({"op": "set", "collection": "photos", "doc_id": x['id'], "file_path": p})
for f in glob.glob(qdir + '/*.json'):
    m = json.load(open(f))
    d = {"title": m["title"], "price": m.get("price"), "url": m.get("url") or f"https://jp.mercari.com/item/{m['id']}",
         "condition": m.get("condition"), "status": "candidate" if m.get("candidate") else "accepted", "createdAt": base}
    p = os.path.abspath(f"docs/L_{m['id']}.json"); json.dump(d, open(p, 'w'), ensure_ascii=False)
    w.append({"op": "set", "collection": "listings", "doc_id": m['id'], "file_path": p})
json.dump(w, open('batch.json', 'w')); print(len(w))
