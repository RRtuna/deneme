import os, sys, tempfile, shutil
sys.path.insert(0,"/home/user/deneme/SYZ/ecg_train")
import ecg_app as A
R=[]
def ck(n,c,d=""):
    R.append(c); print("  %s %s%s"%("PASS" if c else "FAIL",n,("  <- "+str(d)[:200]) if d and not c else ""))

tmp=tempfile.mkdtemp()
data=os.path.join(tmp,"data"); pkg=os.path.join(tmp,"pkg",); os.makedirs(data); os.makedirs(os.path.join(pkg,"models"))
for i in range(37): open(os.path.join(data,"R%03d.hea"%i),"w").write("x")
for f in ("make_submission.py","predict.py","manifest.json"): open(os.path.join(pkg,f),"w").write("x")
open(os.path.join(pkg,"models","m.onnx"),"w").write("x")

print("[1] kayit sayma")
n,s=A.count_records(data); ck("37 kayit",n==37,n); ck("ornek id",len(s)==5)
ck("bos klasor 0",A.count_records(tmp+"/yok")[0]==0)

print("\n[2] sure tahmini (olculen hizlardan)")
for sp,exp in (("mp11",6.2),("threads2",20.9),("seri",92.0)):
    got=A.estimate_seconds(750,sp)/60
    ck("750 @ %-9s ~%.1f dk"%(sp,exp), abs(got-exp)<0.6, "%.1f"%got)

print("\n[3] cikti adi -- TEAM_ ikilenmesin")
ck("807466 -> TEAM_807466_FINAL.json", A.output_name("807466")=="TEAM_807466_FINAL.json", A.output_name("807466"))
ck("TEAM_001 -> TEAM_001_FINAL.json",  A.output_name("TEAM_001")=="TEAM_001_FINAL.json", A.output_name("TEAM_001"))

print("\n[4] komut kurulumu")
c=A.build_command(pkg,data,"tkt-26","807466","4997133","mp11")
ck("--threads 1 var","--threads" in c and c[c.index("--threads")+1]=="1")
ck("--model-parallel 11 var","--model-parallel" in c and c[c.index("--model-parallel")+1]=="11")
ck("--ids YOK (varsayilan)","--ids" not in c, c)
c2=A.build_command(pkg,data,"tkt-26","807466","4997133","mp11",ids_file="x.txt")
ck("--ids istenince var","--ids" in c2)
c3=A.build_command(pkg,data,"t","1","2","seri")
ck("seri modda hiz bayragi yok", "--threads" not in c3 and "--model-parallel" not in c3, c3)

print("\n[5] hazirlik kontrolu -- engelleyiciler")
ck("hazir paket+veri -> sorun yok", A.check_ready(pkg,data,"807466","4997133")==[], A.check_ready(pkg,data,"807466","4997133"))
ck("bos team_id yakalandi", any("team_id" in p for p in A.check_ready(pkg,data,"","4997133")))
_e=os.path.join(tmp,"bos"); os.makedirs(_e,exist_ok=True)
ck("bos klasor yakalandi", any(".hea" in p for p in A.check_ready(pkg,_e,"1","2")), A.check_ready(pkg,_e,"1","2"))
ck("eksik paket yakalandi", any("make_submission" in p or "klasoru secilmedi" in p for p in A.check_ready(tmp,data,"1","2")))

print("\n[6] ilerleme ayristirma (make_submission ciktisindan)")
ck("tahmin 300/750 -> 0.40", abs(A.parse_progress("  tahmin 300/750  120 sn",750)-0.4)<1e-9)
ck("on isleme 100/750",     abs(A.parse_progress("  on isleme 100/750  9 sn",750)-(100/750))<1e-9)
ck("alakasiz satir -> None", A.parse_progress("all checks passed",750) is None)
ck("bozuk satir -> None",    A.parse_progress("  tahmin abc",750) is None)

print("\n[7] CMD alintilama")
q=A.quote_for_cmd(["py","--root",r"C:\Bir Yer\SYZ"])
ck("bosluklu yol tirnaklandi", '"C:\\Bir Yer\\SYZ"' in q, q)

print("\n[8] sonuc okuma")
import json
j=os.path.join(tmp,"o.json")
json.dump({"predictions":[{"id":"a","predicted_class":"AFIB"},{"id":"b","predicted_class":"AFL"}]},open(j,"w"))
n,d,e=A.read_result(j)
ck("2 kayit",n==2 and e is None); ck("dagilim",d["AFIB"]==1 and d["AFL"]==1,d)
n,d,e=A.read_result(os.path.join(tmp,"yok.json")); ck("olmayan dosya -> hata",e is not None)

shutil.rmtree(tmp,True)
print("\n"+"="*50); print("%d/%d gecti"%(sum(R),len(R)))
sys.exit(0 if all(R) else 1)
