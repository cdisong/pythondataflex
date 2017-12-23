import pandas as pd 


jan2017 = "/Users/christine/Desktop/flights/2017/jan2017.csv"
feb2017 = "/Users/christine/Desktop/flights/2017/feb2017.csv"
mar2017 = "/Users/christine/Desktop/flights/2017/mar2017.csv"
apr2017 = "/Users/christine/Desktop/flights/2017/apr2017.csv"
may2017 = "/Users/christine/Desktop/flights/2017/may2017.csv"
jun2017 = "/Users/christine/Desktop/flights/2017/jun2017.csv"
jul2017 = "/Users/christine/Desktop/flights/2017/jul2017.csv"
aug2017 = "/Users/christine/Desktop/flights/2017/aug2017.csv"
sep2017 = "/Users/christine/Desktop/flights/2017/sep2017.csv"


database = pd.read_csv(jan2017, low_memory=False)

first = database.head(10) 
last = database.tail(10)
mid = database.tail(10)
ex = database.head(10)

last = last.reset_index(drop=True) 

vertical = pd.concat([first, last, mid, ex], axis=0, ignore_index=True)
vertical.head(20)

keep_col = ['FL_DATE', 'ORIGIN', 'DEST'] 


new_v = vertical[keep_col] 

new_v.head(20) 


##### 
import pandas as pd 

jan2017 = "/Users/christine/Desktop/flights/2017/jan2017.csv"
feb2017 = "/Users/christine/Desktop/flights/2017/feb2017.csv"
mar2017 = "/Users/christine/Desktop/flights/2017/mar2017.csv"
apr2017 = "/Users/christine/Desktop/flights/2017/apr2017.csv"
may2017 = "/Users/christine/Desktop/flights/2017/may2017.csv"
jun2017 = "/Users/christine/Desktop/flights/2017/jun2017.csv"
jul2017 = "/Users/christine/Desktop/flights/2017/jul2017.csv"
aug2017 = "/Users/christine/Desktop/flights/2017/aug2017.csv"
sep2017 = "/Users/christine/Desktop/flights/2017/sep2017.csv"

jan2016 = "/Users/christine/Desktop/flights/2016/jan2016.csv"
feb2016 = "/Users/christine/Desktop/flights/2016/feb2016.csv"
mar2016 = "/Users/christine/Desktop/flights/2016/mar2016.csv"
apr2016 = "/Users/christine/Desktop/flights/2016/apr2016.csv"
may2016 = "/Users/christine/Desktop/flights/2016/may2016.csv"
jun2016 = "/Users/christine/Desktop/flights/2016/jun2016.csv"
jul2016 = "/Users/christine/Desktop/flights/2016/jul2016.csv"
aug2016 = "/Users/christine/Desktop/flights/2016/aug2016.csv"
sep2016 = "/Users/christine/Desktop/flights/2016/sep2016.csv"
oct2016 = "/Users/christine/Desktop/flights/2016/sep2016.csv"
nov2016 = "/Users/christine/Desktop/flights/2016/sep2016.csv"
dec2016 = "/Users/christine/Desktop/flights/2016/sep2016.csv"

jan2015 = "/Users/christine/Desktop/flights/2015/jan2015.csv"
feb2015 = "/Users/christine/Desktop/flights/2015/feb2015.csv"
mar2015 = "/Users/christine/Desktop/flights/2015/mar2015.csv"
apr2015 = "/Users/christine/Desktop/flights/2015/apr2015.csv"
may2015 = "/Users/christine/Desktop/flights/2015/may2015.csv"
jun2015 = "/Users/christine/Desktop/flights/2015/jun2015.csv"
jul2015 = "/Users/christine/Desktop/flights/2015/jul2015.csv"
aug2015 = "/Users/christine/Desktop/flights/2015/aug2015.csv"
sep2015 = "/Users/christine/Desktop/flights/2015/sep2015.csv"
oct2015 = "/Users/christine/Desktop/flights/2015/sep2015.csv"
nov2015 = "/Users/christine/Desktop/flights/2015/sep2015.csv"
dec2015 = "/Users/christine/Desktop/flights/2015/sep2015.csv"

jan2014 = "/Users/christine/Desktop/flights/2014/jan2014.csv"
feb2014 = "/Users/christine/Desktop/flights/2014/feb2014.csv"
mar2014 = "/Users/christine/Desktop/flights/2014/mar2014.csv"
apr2014 = "/Users/christine/Desktop/flights/2014/apr2014.csv"
may2014 = "/Users/christine/Desktop/flights/2014/may2014.csv"
jun2014 = "/Users/christine/Desktop/flights/2014/jun2014.csv"
jul2014 = "/Users/christine/Desktop/flights/2014/jul2014.csv"
aug2014 = "/Users/christine/Desktop/flights/2014/aug2014.csv"
sep2014 = "/Users/christine/Desktop/flights/2014/sep2014.csv"
oct2014 = "/Users/christine/Desktop/flights/2014/sep2014.csv"
nov2014 = "/Users/christine/Desktop/flights/2014/sep2014.csv"
dec2014 = "/Users/christine/Desktop/flights/2014/sep2014.csv"
 


jan2013 = "/Users/christine/Desktop/flights/2013/jan2013.csv"
feb2013 = "/Users/christine/Desktop/flights/2013/feb2013.csv"
mar2013 = "/Users/christine/Desktop/flights/2013/mar2013.csv"
apr2013 = "/Users/christine/Desktop/flights/2013/apr2013.csv"
may2013 = "/Users/christine/Desktop/flights/2013/may2013.csv"
jun2013 = "/Users/christine/Desktop/flights/2013/jun2013.csv"
jul2013 = "/Users/christine/Desktop/flights/2013/jul2013.csv"
aug2013 = "/Users/christine/Desktop/flights/2013/aug2013.csv"
sep2013 = "/Users/christine/Desktop/flights/2013/sep2013.csv"
oct2013 = "/Users/christine/Desktop/flights/2013/sep2013.csv"
nov2013 = "/Users/christine/Desktop/flights/2013/sep2013.csv"
dec2013 = "/Users/christine/Desktop/flights/2013/sep2013.csv"

a = pd.read_csv(jan2017, low_memory=False) 
b = pd.read_csv(feb2017, low_memory=False)
b = b.reset_index(drop=True) 

c = pd.read_csv(mar2017, low_memory=False) 
c = c.reset_index(drop=True) 

d = pd.read_csv(apr2017, low_memory=False) 
d = d.reset_index(drop=True) 

e = pd.read_csv(may2017, low_memory=False) 
e = e.reset_index(drop=True) 

f = pd.read_csv(jun2017, low_memory=False)
f = f.reset_index(drop=True) 
 
g = pd.read_csv(jul2017, low_memory=False) 
g = g.reset_index(drop=True) 

h = pd.read_csv(aug2017, low_memory=False) 
h = h.reset_index(drop=True) 

i = pd.read_csv(sep2017, low_memory=False) 
i = i.reset_index(drop=True) 

j = pd.read_csv(jan2016, low_memory=False)
j = j.reset_index(drop=True) 

k = pd.read_csv(feb2016, low_memory=False) 
k = k.reset_index(drop=True) 

l = pd.read_csv(mar2016, low_memory=False) 
l = l.reset_index(drop=True) 

m = pd.read_csv(apr2016, low_memory=False) 
m = m.reset_index(drop=True) 

n = pd.read_csv(may2016, low_memory=False)
n = n.reset_index(drop=True) 
 
o = pd.read_csv(jun2016, low_memory=False) 
o = o.reset_index(drop=True) 

p = pd.read_csv(jul2016, low_memory=False) 
p = p.reset_index(drop=True) 

q = pd.read_csv(aug2016, low_memory=False) 
q = q.reset_index(drop=True) 

r = pd.read_csv(sep2016, low_memory=False)
r = r.reset_index(drop=True) 

s = pd.read_csv(oct2016, low_memory=False) 
s = s.reset_index(drop=True) 

t = pd.read_csv(nov2016, low_memory=False) 
t = t.reset_index(drop=True) 

u = pd.read_csv(dec2016, low_memory=False) 
u = u.reset_index(drop=True) 

v = pd.read_csv(jan2015, low_memory=False)
v = v.reset_index(drop=True) 
 
w = pd.read_csv(feb2015, low_memory=False) 
w = w.reset_index(drop=True) 

x = pd.read_csv(mar2015, low_memory=False) 
x = x.reset_index(drop=True) 

y = pd.read_csv(apr2015, low_memory=False) 
y = y.reset_index(drop=True) 

z = pd.read_csv(may2015, low_memory=False)
z = z.reset_index(drop=True) 

aa = pd.read_csv(jun2015, low_memory=False) 
aa = aa.reset_index(drop=True) 

bb = pd.read_csv(jul2015, low_memory=False) 
bb = bb.reset_index(drop=True) 

cc = pd.read_csv(aug2015, low_memory=False) 
cc = cc.reset_index(drop=True) 

dd = pd.read_csv(sep2015, low_memory=False)
dd = dd.reset_index(drop=True) 
 
ee = pd.read_csv(oct2015, low_memory=False) 
ee = ee.reset_index(drop=True) 

ff = pd.read_csv(nov2015, low_memory=False) 
ff = ff.reset_index(drop=True) 

gg = pd.read_csv(dec2015, low_memory=False) 
gg = gg.reset_index(drop=True) 

hh = pd.read_csv(jan2014, low_memory=False)
hh = hh.reset_index(drop=True) 

ii = pd.read_csv(feb2014, low_memory=False) 
ii = ii.reset_index(drop=True) 

jj = pd.read_csv(mar2014, low_memory=False) 
jj = jj.reset_index(drop=True) 

kk = pd.read_csv(apr2014, low_memory=False) 
kk = kk.reset_index(drop=True) 

ll = pd.read_csv(may2014, low_memory=False)
ll = ll.reset_index(drop=True) 
 
mm = pd.read_csv(jun2014, low_memory=False) 
mm = mm.reset_index(drop=True) 

nn = pd.read_csv(jul2014, low_memory=False) 
nn = nn.reset_index(drop=True) 

oo = pd.read_csv(aug2014, low_memory=False) 
oo = oo.reset_index(drop=True) 

pp = pd.read_csv(sep2014, low_memory=False)
pp = pp.reset_index(drop=True) 

qq = pd.read_csv(oct2014, low_memory=False) 
qq = qq.reset_index(drop=True) 

rr = pd.read_csv(nov2014, low_memory=False) 
rr = rr.reset_index(drop=True) 

ss = pd.read_csv(dec2014, low_memory=False) 
ss = ss.reset_index(drop=True) 

tt = pd.read_csv(jan2013, low_memory=False)
tt = tt.reset_index(drop=True) 
 
uu = pd.read_csv(feb2013, low_memory=False) 
uu = uu.reset_index(drop=True) 

vv = pd.read_csv(mar2013, low_memory=False) 
vv = vv.reset_index(drop=True) 

ww = pd.read_csv(apr2013, low_memory=False) 
ww = ww.reset_index(drop=True)

xx = pd.read_csv(may2013, low_memory=False)
xx = xx.reset_index(drop=True) 

yy = pd.read_csv(jun2013, low_memory=False) 
yy = yy.reset_index(drop=True) 

zz = pd.read_csv(jul2013, low_memory=False) 
zz = zz.reset_index(drop=True) 

aaa = pd.read_csv(aug2013, low_memory=False) 
aaa = aaa.reset_index(drop=True) 

bbb = pd.read_csv(sep2013, low_memory=False)
bbb = bbb.reset_index(drop=True) 
 
ccc = pd.read_csv(oct2013, low_memory=False) 
ccc = ccc.reset_index(drop=True) 

ddd = pd.read_csv(nov2013, low_memory=False) 
ddd = ddd.reset_index(drop=True) 

eee = pd.read_csv(dec2013, low_memory=False) 
eee = eee.reset_index(drop=True) 


twenty17 = pd.concat([a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z, aa, bb, cc, dd, ee, ff, gg, hh, ii, jj, kk, ll, mm, nn, oo, pp, qq, rr, ss, tt, uu, vv, ww, xx, yy, zz, aaa, bbb, ccc, ddd, eee], axis=0, ignore_index=True)

twenty17.shape 

###works up to here fuck yes 



keep_col = ['FL_DATE', 'ORIGIN', 'DEST', 'CRS_ELAPSED_TIME', 'ACTUAL_ELAPSED_TIME']
new_v = twenty17[keep_col] 


# works but the FL_DATE from head to tail gets changed so i have to normalize that in a bit. new