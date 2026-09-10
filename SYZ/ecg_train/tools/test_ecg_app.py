"""ecg_app cekirdek testleri -- tkinter GEREKMEZ."""
import os, sys, json, tempfile, shutil
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ecg_app as A

R=[]
def ck(n,c,d=""):
    R.append(bool(c)); print("  %s %s%s"%("PASS" if c else "FAIL",n,("  <- "+str(d)[:220]) if d and not c else ""))

print("[1] min/max zarfi -- QRS tepesi KAYBOLMAMALI")
# 5000 ornek, tek bir keskin R tepesi. Duz seyreltme onu kacirabilir.
y=np.zeros(5000); y[2501]=4.0
lo,hi=A.envelope(y,900)
ck("tepe korundu (max=4.0)", abs(hi.max()-4.0)<1e-9, hi.max())
ck("sutun sayisi 900", lo.size==900 and hi.size==900)
ck("duz seyreltme kacirirdi", y[::5][:900].max()<4.0-1e-9, "kontrol")
neg=np.zeros(5000); neg[1234]=-3.0
ck("negatif tepe de korundu", abs(A.envelope(neg,600)[0].min()+3.0)<1e-9)
ck("kisa sinyal aynen doner", np.array_equal(A.envelope(np.arange(5.),10)[0], np.arange(5.)))
ck("bos sinyal cokmez", A.envelope(np.zeros(0),10)[0].size==0)

print("\n[2] ortak olcek -- derivasyonlar arasi oran korunmali")
sig=np.zeros((12,1000)); sig[0]=1.0; sig[1]=2.0     # II, I'in iki kati
sc=A.trace_scale(sig,12,40.0)
ck("olcek pozitif", sc>0)
ck("en buyuk genlik tasmiyor", 2.0*sc <= 40.0*0.42+1e-9, 2.0*sc)
ck("duz sinyalde cokmez", A.trace_scale(np.zeros((12,10)),12,40.0)==1.0)

print("\n[3] yerlesim")
tr=A.layout_traces(np.random.RandomState(0).randn(12,5000)*0.5, 900, 480)
ck("12 derivasyon", len(tr)==12, len(tr))
ck("her derivasyonun noktalari var", all(len(c)>100 for _i,_b,c in tr))
bl=[b for _i,b,_c in tr]
ck("taban cizgileri artan", all(bl[i]<bl[i+1] for i in range(11)))
ck("taban cizgileri tuvalde", all(0<b<480 for b in bl))
ck("kucuk tuvalde cokmez", A.layout_traces(np.zeros((12,100)),20,20)==[])

print("\n[4] olasilik siralama ve guven")
p=np.array([0.05,0.62,0.21,0.08,0.04])
rows=A.sort_probs(p)
ck("en yuksek AFIB", rows[0][0]=="AFIB" and abs(rows[0][1]-0.62)<1e-9, rows[0])
ck("azalan sirali", all(rows[i][1]>=rows[i+1][1] for i in range(4)))
ck("guven YUKSEK", A.confidence_label(np.array([0.9,.04,.03,.02,.01]))[0]=="YUKSEK")
ck("guven ORTA",   A.confidence_label(np.array([0.6,.2,.1,.05,.05]))[0]=="ORTA")
ck("guven DUSUK",  A.confidence_label(np.array([0.4,.35,.15,.05,.05]))[0].startswith("DUSUK"))

print("\n[5] olcum secimi")
import ecg_preprocess as ep
names=list(ep.FEATURE_NAMES); f=np.arange(len(names),dtype=float)
v=A.pick_features(f,names)
ck("bazi olcumler secildi", len(v)>=3, len(v))
ck("etiket+deger cifti", all(len(t)==2 for t in v))
ck("NaN atlanir", len(A.pick_features(np.full(len(names),np.nan),names))==0)
ck("None guvenli", A.pick_features(None,names)==[])

print("\n[6] toplu komut")
c=A.build_batch_command("/pkg","/data","tkt-26","807466","4997133","mp11")
ck("--model-parallel 11","--model-parallel" in c and c[c.index("--model-parallel")+1]=="11")
ck("--ids yok (varsayilan)","--ids" not in c)
ck("seri modda bayrak yok", "--threads" not in A.build_batch_command("/p","/d","a","1","2","seri"))
ck("cikti adi", A.output_name("807466")=="TEAM_807466_FINAL.json")
ck("TEAM_ ikilenmez", A.output_name("TEAM_9")=="TEAM_9_FINAL.json")

print("\n[6b] CLI yetenek algilama -- YASANAN HATA")
# Kullanicinin gordugu hata: eski competition_package'a --model-parallel 11
# gonderildi -> "unrecognized arguments". Bu bir daha URETILEMEMELI.
ESKI = {"--root","--ids","--models","--threads","--team-name","--team-id",
        "--application-id","--out","--max-fail","--validate","--workers",
        "--cache-sessions","--retag"}
YENI = ESKI | {"--model-parallel"}
ck("eski CLI'da --model-parallel yok", "--model-parallel" not in ESKI)
c=A.build_batch_command("/pkg","/d","t","1","2","mp11",caps=ESKI)
ck("eski betige --model-parallel GONDERILMEZ", "--model-parallel" not in c, c)
ck("eski betikte en iyi desteklenen: threads2",
   "--threads" in c and c[c.index("--threads")+1]=="2", c)
ck("dusurme sebebi bildirilir",
   "--model-parallel" in A.choose_preset("mp11",ESKI)[1], A.choose_preset("mp11",ESKI))
c=A.build_batch_command("/pkg","/d","t","1","2","mp11",caps=YENI)
ck("yeni betikte MP11 korunur",
   "--model-parallel" in c and c[c.index("--model-parallel")+1]=="11", c)
ck("yeni betikte dusurme yok", A.choose_preset("mp11",YENI)[1]=="")
ck("caps=None -> eski davranis (filtre yok)",
   "--model-parallel" in A.build_batch_command("/p","/d","t","1","2","mp11"))
ck("--ids desteklenmiyorsa atlanir",
   "--ids" not in A.build_batch_command("/p","/d","t","1","2","seri",
                                        ids_file="x.txt",
                                        caps={"--root","--team-name",
                                              "--team-id","--application-id"}))
ck("--ids destekleniyorsa eklenir",
   "--ids" in A.build_batch_command("/p","/d","t","1","2","seri",
                                    ids_file="x.txt",caps=ESKI))
try:
    A.build_batch_command("/p","/d","t","1","2","seri",caps={"--root"})
    ck("temel bayrak eksikse ValueError", False, "hata atmadi")
except ValueError as e:
    ck("temel bayrak eksikse ValueError", "--team-id" in str(e), e)
ck("bos caps'te seri secilir", A.choose_preset("mp11",set())[0]=="seri")

print("\n[6c] yetenek ayristirma")
H=("usage: make_submission.py [-h] [--root ROOT] [--threads THREADS]\n"
   "  --cache-sessions   onbellek\n  --model-parallel N  paralel\n")
f=A.parse_supported_flags(H)
ck("--help'ten cikarilir", {"--root","--threads","--model-parallel"} <= f, sorted(f))
ck("-h uzun secenek degil", "-h" not in f)
ck("bos metin -> bos kume", A.parse_supported_flags("")==set())
t=tempfile.mkdtemp()
sp=os.path.join(t,"make_submission.py")
open(sp,"w").write('ap.add_argument("--root")\nap.add_argument("--model-parallel", type=int)\n')
ck("kaynak taramasi calisir",
   A.scan_script_flags(sp)=={"--root","--model-parallel"}, A.scan_script_flags(sp))
ck("olmayan dosya -> bos", A.scan_script_flags(os.path.join(t,"yok.py"))==set())
cap=A.probe_capabilities(os.path.join(t,"yok.py"))
ck("olmayan betik: bos yetenek + hata", cap["flags"]==set() and cap["error"])
ck("script_path birlestirir",
   A.script_path("/pkg").replace("\\","/")=="/pkg/make_submission.py")
shutil.rmtree(t,True)

print("\n[6d] paket secimi -- MP destekleyen TERCIH EDILIR")
t=tempfile.mkdtemp()
def mkpkg(d, mp):
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d,"predict.py"),"w").write("x")
    open(os.path.join(d,"make_submission.py"),"w").write(
        'ap.add_argument("--root")\nap.add_argument("--threads")\n'
        + ('ap.add_argument("--model-parallel")\n' if mp else ""))
mkpkg(os.path.join(t,"competition_package"), False)   # eski -- MP YOK
mkpkg(os.path.join(t,"release","verified_mp11_fast_2026-09-09","package"), True)
sel,infos=A.resolve_package(t)
ck("MP'li release secildi", "verified_mp11_fast" in sel, sel)
ck("adaylar listelendi", len(infos)>=2, len(infos))
shutil.rmtree(t,True)
t=tempfile.mkdtemp()
mkpkg(os.path.join(t,"competition_package"), False)   # tek aday, MP yok
sel,_=A.resolve_package(t)
ck("MP yoksa mevcut paket secilir", sel.endswith("competition_package"), sel)
shutil.rmtree(t,True)

print("\n[7] ilerleme ayristirma")
ck("300/750 -> 0.40", abs(A.parse_progress("  tahmin 300/750  12 sn",750)-0.4)<1e-9)
ck("alakasiz -> None", A.parse_progress("all checks passed",750) is None)

print("\n[8] kayit tarama")
t=tempfile.mkdtemp()
os.makedirs(os.path.join(t,"alt"))
for n in ("b","a"): open(os.path.join(t,n+".hea"),"w").write("x")
open(os.path.join(t,"alt","c.hea"),"w").write("x")
recs=A.find_records(t)
ck("3 kayit, alt klasor dahil", len(recs)==3, len(recs))
ck("id'ye gore sirali", [r[0] for r in recs]==["a","b","c"], [r[0] for r in recs])
ck("olmayan klasor -> bos", A.find_records(t+"/yok")==[])
shutil.rmtree(t,True)

print("\n[9] sure tahmini")
ck("750 @ mp11 ~6 dk", abs(A.estimate_seconds(750,"mp11")/60-6.2)<0.4)
ck("bicimleme", A.human_time(45)=="45 sn" and "dk" in A.human_time(400))

print("\n"+"="*54); print("%d/%d gecti"%(sum(R),len(R)))
sys.exit(0 if all(R) else 1)
