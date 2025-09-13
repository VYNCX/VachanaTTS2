# เครดิต : TLTK และ https://github.com/nozomiyamada/thaig2p มีการดัดแปลงและลบส่วนที่ไม่จำเป็นออก

import csv
import re
import math
import os
from copy import deepcopy
from collections import defaultdict
import pickle

def read_thdict(Filename):
    global TDICT
    fileObject = open(Filename,'rb')  
    TDICT = pickle.load(fileObject)

def tltkg2p(Input):
    global SegSep
    global SSegSep
    output = ""
    out = ""
    
    Input = preprocess(Input)
    sentLst = Input.split(SegSep)
    for s in sentLst:
        inLst = s.split(SSegSep)
        for inp in inLst:
            if inp == '': continue            
            objMatch = re.match(r"[^ก-์]+",inp)
            if objMatch:
                out = inp+'<tr/>'+inp
            else:
                y = sylparse(inp)
                out = wordparse(y)
            output = output+out+WordSep
        output = output+'<s/>' 
    return(output)        

def sylparse(Input):
    global SylSep
    global PRON
    global PRONUN
    
    PRONUN = defaultdict(list)
    schart = defaultdict(dict)
    probEnd = defaultdict(float)
    schartx = {}
    schart.clear()
    probEnd.clear()
    tmp = []
    
    EndOfInput = len(Input)
    for f in PRON:
        for i in range(EndOfInput):
            Inx = Input[i:]
            matchObj = re.match(f,Inx)
            if matchObj:
                keymatch = matchObj.group()
                try:
                    matchObj.group(3)
                    charmatch = matchObj.group(1) + ' ' + matchObj.group(2) + ' ' + matchObj.group(3)
                except IndexError:
                    try:
                        matchObj.group(2)
                        charmatch = matchObj.group(1) + ' ' + matchObj.group(2) 
                    except IndexError:
                        try:
                            matchObj.group(1)
                            charmatch = matchObj.group(1) 
                        except IndexError:
                            PRONUN[matchObj.group()].extend(PRON[f])
                k=i+len(matchObj.group())
                schart[i][k] = [matchObj.group()]

                for PronF in PRON[f]:

                    codematch = re.sub(r"[^AKYDZCRX]","",PronF)
                    if codematch:
                        phone = ReplaceSnd(PronF,codematch,charmatch)
                        if  NotExceptionSyl(codematch,charmatch,keymatch,phone):
                            (phone,tone) = ToneAssign(keymatch,phone,codematch,charmatch)
                            (keymatch,phone) = TransformSyl(keymatch,phone)         
                            if (tone < '5'): 
                                phone = re.sub(r'8',tone,phone)          
                                PRONUN[''.join(schart[i][k])].append(phone)
                        if  re.match(r'ทร',keymatch)  and  re.match(r"thr",phone):
                            phone=re.sub(r"thr","s",phone) 
                            PRONUN[''.join(schart[i][k])].append(phone)
                    else:
                        PRONUN[''.join(schart[i][k])].append(PronF)

                    probEnd[(i,k)] = prob_trisyl(schart[i][k])

    for j in range(EndOfInput):
        schartx = deepcopy(schart)
        if j in schart[0]:
            s1 = schart[0][j]
            for k in schart[j]:
                s2 = schart[j][k]
                tmp = mergekaran1(s1+s2)
                if k not in schart[0]:                        
                    schartx[0][k] = tmp
                    probEnd[(0,k)] = prob_trisyl(tmp)
                else:
                    p = prob_trisyl(tmp)
                    if p > probEnd[(0,k)]:
                        schartx[0][k] = tmp 
                        probEnd[(0,k)] = p

        schart = deepcopy(schartx)
    if EndOfInput in schart[0]:    
        return(SylSep.join(schart[0][EndOfInput]))
    else:
        return('<Fail>'+Input+'</Fail>')

def ReplaceSnd(phone,codematch,charmatch):
     global stable
     snd = phone
     tmp1Lst = charmatch.split(' ')   #get character
     i=0
     for x in list(codematch):
          s = stable[x][tmp1Lst[i]]
          snd = re.sub(x,s,snd)
          i += 1 
     snd += '8'
     return(snd)

def NotExceptionSyl(codematch,charmatch,form,phone):
    if re.search(r'\.',form):  return(True)
    if 'CR' in codematch:        
        if re.match(r'ผ ร|ด ล|ต ล|ท ล|ด ว|ต ว|ท ว|บ ว|ป ว|พ ว|ฟ ว|ผ ว|ส ล|ส ว|ร ร|ศ ล|ศ ว',charmatch):  return(False)
    if 'AK' in codematch: 
        clst = charmatch.split(' ')
        if clst[1] not in AK[clst[0]]: return(False)
    if re.search(r'\u0E31[\0E48-\u0E4B]?ว]',form) and 'aw' in phone: return(False)
    if re.search(r'[ก-ฮ] ข',charmatch) and not re.search(r'[\u0E38\u0E31\u0E40]',form): return(False)
    return(True)

def ToneAssign(keymatch,phone,codematch,charmatch):
    if phone == '' : return('','9')
    lead = ''
    init = ''
    final = ''
    if re.search(r'[0-4]8',phone):
        phone = re.sub(r'([0-4])8',r'\1',phone)
        return(phone,'')
    if 'X' in codematch or codematch == 'GH' or codematch == 'EF':
        lx = charmatch.split(' ')
        lead = ''
        init = lx[0]
        if len(lx) > 1:
            final = lx[1]
        else: final = ''    
    elif re.search(r'AK',codematch) or re.search(r'CR',codematch):
        lx = charmatch.split(' ')
        lead = lx[0]
        if len(lx) > 2:
            final = lx[2]
            init = lx[1]
        elif len(lx) >1:    
            init = lx[1]
            final = ''

    deadsyll = DeadSyl(phone)

    if "+'" in phone:
        if lead in 'ผฝถขสหฉศษ':
            phone = re.sub(r'\+','1',phone)
        elif lead in 'กจดตบปอ':
            phone = re.sub(r'\+','1',phone)
        else:    
            phone = re.sub(r'\+','3',phone)

    if init in 'กจดตฎฏบปอ':   # middle consonant
        if deadsyll == 'L':
            if re.search(r'\u0E48',keymatch): return(phone,'1')
            elif re.search(r'\u0E49',keymatch): return(phone,'2')
            elif re.search(r'\u0E4A',keymatch): return(phone,'3')
            elif re.search(r'\u0E4B',keymatch): return(phone,'4')
            else: return(phone,'0')
        else:
            if re.search(r'\u0E48',keymatch): return(phone,'9')
            elif re.search(r'\u0E49',keymatch): return(phone,'2')
            elif re.search(r'\u0E4A',keymatch): return(phone,'3')
            elif re.search(r'\u0E4B',keymatch): return(phone,'4')
            else: return(phone,'1')
    elif init in 'ขฃฉฐถผฝสศษห':   # high consonant
        if deadsyll == 'L':
            if re.search(r'\u0E48',keymatch): return(phone,'1')
            elif re.search(r'\u0E49',keymatch): return(phone,'2')
            elif re.search(r'\u0E4A',keymatch): return(phone,'9')
            elif re.search(r'\u0E4B',keymatch): return(phone,'9')
            else: return(phone,'4')
        else:
            if re.search(r'\u0E48',keymatch): return(phone,'9')
            elif re.search(r'\u0E49',keymatch): return(phone,'2')
            elif re.search(r'\u0E4A',keymatch): return(phone,'9')
            elif re.search(r'\u0E4B',keymatch): return(phone,'9')
            else: return(phone,'1')
    elif init in 'งญณนมยรลวฬ' and lead != '' and lead in 'ขฃฉฐถผฝสศษห':
        if deadsyll == 'L':
            if re.search(r'\u0E48',keymatch): return(phone,'1')
            elif re.search(r'\u0E49',keymatch): return(phone,'2')
            elif re.search(r'\u0E4A',keymatch): return(phone,'9')
            elif re.search(r'\u0E4B',keymatch): return(phone,'9')
            else: return(phone,'4')
        else:
            if re.search(r'\u0E48',keymatch): return(phone,'9')
            elif re.search(r'\u0E49',keymatch): return(phone,'2')
            elif re.search(r'\u0E4A',keymatch): return(phone,'9')
            elif re.search(r'\u0E4B',keymatch): return(phone,'9')
            else: return(phone,'1')
    elif init in 'งญณนมยรลวฬ' and lead != '' and lead in 'กจดตฎฏบปอ':
        if deadsyll == 'L':
            if re.search(r'\u0E48',keymatch): return(phone,'1')
            elif re.search(r'\u0E49',keymatch): return(phone,'2')
            elif re.search(r'\u0E4A',keymatch): return(phone,'3')
            elif re.search(r'\u0E4B',keymatch): return(phone,'4')
            else: return(phone,'0')
        else:
            if re.search(r'\u0E48',keymatch): return(phone,'9')
            elif re.search(r'\u0E49',keymatch): return(phone,'2')
            elif re.search(r'\u0E4A',keymatch): return(phone,'3')
            elif re.search(r'\u0E4B',keymatch): return(phone,'4')
            else: return(phone,'1')
    elif init in 'คฅฆชฌซฑฒทธพภฟฮงญณนมยรลวฬฤฦ':
        if deadsyll == 'L':
            if re.search(r'\u0E48',keymatch): return(phone,'2')
            elif re.search(r'\u0E49',keymatch): return(phone,'3')
            elif re.search(r'\u0E4A',keymatch): return(phone,'9')
            elif re.search(r'\u0E4B',keymatch): return(phone,'9')
            else: return(phone,'0')        
        elif re.search(r'[aeiouxOU\@][aeiouxOU\@]+',phone):
            if re.search(r'\u0E48',keymatch): return(phone,'9')
            elif re.search(r'\u0E49',keymatch): return(phone,'3')
            elif re.search(r'\u0E4A',keymatch): return(phone,'9')
            elif re.search(r'\u0E4B',keymatch): return(phone,'4')
            else: return(phone,'2')
        else:
            if re.search(r'\u0E48',keymatch): return(phone,'2')
            elif re.search(r'\u0E49',keymatch): return(phone,'9')
            elif re.search(r'\u0E4A',keymatch): return(phone,'9')
            elif re.search(r'\u0E4B',keymatch): return(phone,'4')
            else: return(phone,'3')

def DeadSyl(phone):
    inx = phone
    inx = re.sub('ch','C',inx)
    inx = re.sub(r'[0-4]','',inx)
    if re.search(r'[mnwjlN]8?$',inx):
        return('L')
    elif re.search(r'[pktfscC]8?$',inx):
        return('D')
    elif re.search(r'([aeiouxOU\@])\1',inx):
        return('L')
    else:
        return('D')

def TransformSyl(form,phone):
    if re.search(r'xx[nmN][12]',phone):
        phone = re.sub(r'xx','x',phone)
    elif re.search(r'ee[nmN][12]',phone):
        phone = re.sub(r'ee','e',phone)
    elif re.search(r'\@\@[nmN][12]',phone):
        phone = re.sub(r'\@\@','\@',phone)
    if re.search(r'^อย่า$|^อยู่$|^อย่าง$|^อยาก$',form) and "'" in phone:
        x = phone.split("'")
        phone = x[-1]
    elif 'ห' in form and 'ha1' in phone and not re.search(r'หนุ$|หก|หท|หพ|หฤ|หโ',form):
        x = phone.split("'")
        phone = x[-1]
    elif re.search(r'[จซศส]ร',form) and re.search(r'[cs]r',phone) and re.search(r"[^']",phone):
        phone = re.sub('r','',phone)
    return (form,phone)

def wordparse(Input):
    global TDICT
    global EndOfSent
    global chart
    global SegSep
    global WordSep
    global CollocSt
    
    maiyamok_find = r"(<tr/>|\|)" + r"([?a-zENOU0-9~'@^]+?)"  + r"[|~]ๆ"
    maiyamok_rep = r"\1\2" + WordSep + r"\2"

    part = []
    chart = defaultdict(dict)
    SylSep = '~'
    outx = ""
    chart.clear()
    CollocSt = defaultdict(float)
    
    part = Input.split(SegSep)
    for inx in part:
        SylLst = inx.split(SylSep)
        EndOfSent = len(SylLst)
        for i in range(EndOfSent):
            chart[i][i+1] = [SylLst[i]]
        for i in range(EndOfSent):
            for j in range(i,EndOfSent+1):
                wrd = ''.join(SylLst[i:j])
                if wrd in TDICT:
                    chart[i][j] = ['~'.join(SylLst[i:j])]
                    if j > i+1:
                        St = 0.0
                        NoOfSyl = len(SylLst[i:j])
                        for ii in range(i,j-1):
                            St += compute_colloc("mi",SylLst[ii],SylLst[ii+1])
                        CollocSt[(i,j)] = St
                    else:
                        CollocSt[(i,j)] = 0.0
        if chart_parse():
            outx += WordSep.join(chart[0][EndOfSent])
            outx = outx.replace('~ๆ','|ๆ')
            outx += '<tr/>'
            outp = []
            for  wx in chart[0][EndOfSent]:
                tmp = wx.split(SylSep)
                op = SelectPhones(tmp)    
                outp.append(op)
            outx += WordSep.join(outp)
            outx = re.sub(maiyamok_find,maiyamok_rep,outx)        
            return(outx)
        else:
            return("<Fail>"+Input+"</Fail>")

def SelectPhones(slst):
   global PRONUN 
   p=''
   out = []
   prmax = 0.

   slst = ['|'] + slst + ['|']
   i = 1
   for i in range(1,len(slst)-1):
        outp = ''
        prmax = 0.
        if len(PRONUN[slst[i]]) == 1:
            out.append(PRONUN[slst[i]][0])
            continue
        else:
            for p in PRONUN[slst[i]]:
                pr = ProbPhone(p, slst[i-1],slst[i],slst[i+1])
                if pr > prmax:
                   prmax = pr
                   outp = p
                elif pr == prmax:
                   if re.search(r"'",p)  and len(p) > len(outp):
                      prmax = pr
                      outp = p
        out.append(outp)
        i += 1
   return('~'.join(out))

def ProbPhone(p,pw,w,nw):
    global PhSTrigram
    global FrmSTrigram
    global PhSBigram
    global FrmSBigram
    global PhSUnigram
    global FrmSUnigram
    global AbsUnigram
    global AbsFrmSUnigram

    p3=0.
    p2=0.
    p1=0.
    p0=0.
    if PhSTrigram[(pw,w,nw,p)] > 0.:
        p3 = (1. + math.log(PhSTrigram[(pw,w,nw,p)])) / (1. + math.log(FrmSTrigram[(pw,w,nw)]))
    if PhSBigram[(pw,w,p)] > 0.:
        p2 = (1. + math.log(PhSBigram[(pw,w,p)])) / (1. + math.log(FrmSBigram[(pw,w)])) * 0.25
    if PhSBigram[(w,nw,p)] > 0.:
        p2 = p2 + (1. + math.log(PhSBigram[(w,nw,p)])) / (1. + math.log(FrmSBigram[(w,nw)])) * 0.75
    if PhSUnigram[(w,p)] > 0.:
        p1 = (1 + math.log(PhSUnigram[(w,p)])) / (1. + math.log(FrmSUnigram[w]))

    abs_w = re.sub(r"[่้๊๋]","",w)
    abs_w = re.sub(r"[ก-ฮ]","C",abs_w)
    abs_p = re.sub(r"[0-9]","",p)
    abs_p = re.sub(r"[^aeio@OuxU]","C",abs_p)
    if AbsUnigram[(abs_w,abs_p)] > 0.:
        p0 = (1 + math.log(AbsUnigram[(abs_w,abs_p)])) / (1. + math.log(AbsFrmSUnigram[abs_w]))
    prob =  0.8*p3 + 0.16*p2 + 0.03*p1 + 0.00001*p0 + 0.00000000001
    return(prob)

def mergekaran1(Lst):
   rs = []
   global MKaran
   MKaran.clear()
   Found = 'n'
   Lst.reverse()
   for s in Lst:
        if re.search(r"(.+)[ิุ]์",s):
            if len(s) < 4:
                Found = 'y'
                x = s
                continue
        elif  re.search(r"(.+)์",s):
            if len(s) < 4:
                Found = 'y'
                x = s
                continue
        if Found == 'y':
            for ph in PRONUN[s]:
                if (s+x,ph) not in MKaran:
                    PRONUN[s+x].append(ph)
                    MKaran[(s+x,ph)] = 1 
            s += x
            rs.append(s)
            Found = 'n'
        else:
            rs.append(s)
   rs.reverse()
   return(rs)

def prob_trisyl(SylLst):
    global TriCount
    global BiCount
    global Count
    global BiType
    global Type
    global TotalWord
    global TotalLex
    global SegSep
    Prob = defaultdict(float)

    pw2 = SegSep
    pw1 = SegSep
    Probx = 1.0

    for w in SylLst:
        if (w,pw1,pw2) in Prob:
            Proba = Prob[(w,pw1,pw2)]
        else:
            Prob[(w,pw1,pw2)] = prob_wb(w,pw1,pw2)
            Proba = Prob[(w,pw1,pw2)]
        Probx += Proba
        pw2 = pw1
        pw1 = w
    return(Probx)

def prob_wb(w,pw1,pw2):
    global TriCount
    global BiCount
    global Count
    global BiType
    global Type
    global TotalWord
    global TotalLex
    
    p3 = 0.0
    p2 = 0.0
    p1 = 0.0
    p = 0.0
    px1 = 0.0
    
    if TriCount[(pw2,pw1,w)] > 0:
        p3 = float(TriCount[(pw2,pw1,w)]) / float( BiCount[(pw2,pw1)] + BiType[(pw2,pw1)])
    if BiCount[(pw1,w)] > 0:
        p2 = float( BiCount[(pw1,w)]) / float((Count[pw1] + Type[pw1]) )
    if Count[w] > 0:
        p1 = float( Count[w]) / float(TotalWord + TotalLex)
    p = 0.8 * p3 + 0.15 * p2 + 0.04 * p1 + 1.0 / float((TotalWord + TotalLex)**2)
    p = math.log(p)
    return(p)

def chart_parse():
    global chart
    global CollocSt
    
    for j in range(EndOfSent):
        chartx = deepcopy(chart)
        if j in chart[0]:
            s1 = chart[0][j]
            for k in chart[j]:
                s2 = chart[j][k]
                if k not in chart[0]:                        
                    chartx[0][k] = s1+s2
                    CollocSt[(0,k)] = CollocSt[(0,j)] + CollocSt[(j,k)]
                else:
                    if CollocSt[(0,j)]+CollocSt[(j,k)] > CollocSt[(0,k)]:
                        CollocSt[(0,k)] = CollocSt[(0,j)] + CollocSt[(j,k)]
                        chartx[0][k] = s1+s2
        chart = deepcopy(chartx)
    if EndOfSent in chart[0]:
        return(1)
    else:
        return(0)

def compute_colloc(stat,w1,w2):
    global TriCount
    global BiCount
    global Count
    global BiType
    global Type
    global TotalWord
    global TotalLex

    if BiCount[(w1,w2)] < 1 or Count[w1] < 1 or Count[w2] < 1:
        BiCount[(w1,w2)] +=1
        Count[w1] +=1
        Count[w2] +=1 
        TotalWord +=2

    if stat == "mi":
        mi = float(BiCount[(w1,w2)] * TotalWord) / float((Count[w1] * Count[w2]))
        value = abs(math.log(mi,2))
    if stat == "chi2":
        value=0
        O11 = BiCount[(w1,w2)]
        O21 = Count[w2] - BiCount[(w1,w2)]
        O12 = Count[w1] - BiCount[(w1,w2)]
        O22 = TotalWord - Count[w1] - Count[w2] +  BiCount[(w1,w2)]
        value = float(TotalWord * (O11*O22 - O12 * O21)**2) / float((O11+O12)*(O11+O21)*(O12+O22)*(O21+O22))
    return(value)

def read_sylpattern(Filename):
    global PRON
    global stable
    global AK
    global MKaran
    global EngAbbr
    
    stable = defaultdict(defaultdict)
    AK = defaultdict(str)
    MKaran = defaultdict(int)
    
    tmp = [] 
    file1 = open(Filename,'r',encoding = 'cp874')
    for line in file1:
        if re.match(r'#',line):
            continue
        line = line.rstrip()
        tmp = line.split(',')
        tmp[0] = re.sub(r"X",u"([ก-ฮ])",tmp[0])
        tmp[0] = re.sub(r"C",u"([กขคจดตทบปผพฟสศซ])",tmp[0])
        tmp[0] = re.sub(r'Y',u"([ก-ฬฮ])",tmp[0])
        tmp[0] = re.sub(r'R',u"([รลว])",tmp[0])
        tmp[0] = re.sub(r'K',u"([ก-ฮ])",tmp[0])
        tmp[0] = re.sub(r'A',u"([กจฆดตบปอขฉฐถผฝศษสหคชพภทธมยรลวนณซญฑฏฌ])",tmp[0])
        tmp[0] = re.sub(r'Z',u"([กงดนมบรลฎฏจตณถพศทสชคภปญ])",tmp[0])
        tmp[0] = re.sub(r'D',u"([กงดนมบวยต])",tmp[0])
        tmp[0] = re.sub(r'W',u"[ก-ฮ]",tmp[0])
        tmp[0] = re.sub(r'\.',u"[\.]",tmp[0])

        if tmp[2] == "T":
            tmp[0] = re.sub(r"T",u"[่้๊๋]",tmp[0])
        else:
            tmp[0] = re.sub(r"T",u"[่้๊๋]*",tmp[0])
            
        PRON[tmp[0]].append(tmp[1])
    
    stable['X'] = { 'ก' : 'k', 'ข' : 'kh' , 'ฃ':'kh', 'ค' : 'kh', 'ฅ' : 'kh','ฆ' : 'kh', 'ง' : 'N', 'จ' : 'c', 'ฉ' : 'ch', 'ช' : 'ch', 'ซ' : 's', 'ฌ' : 'ch','ญ' : 'j','ฎ' : 'd','ฏ' : 't','ฐ' : 'th','ฑ' : 'th','ฒ' : 'th','ณ' : 'n','ด' : 'd','ต' : 't','ถ' : 'th','ท' : 'th','ธ' : 'th','น' : 'n','บ' : 'b','ป' : 'p','ผ' : 'ph','ฝ' : 'f','พ' : 'ph','ฟ' : 'f','ภ' : 'ph','ม' : 'm','ย' : 'j','ร' : 'r','ฤ' : 'r','ล' : 'l','ฦ' : 'l','ว' : 'w','ศ' : 's','ษ' : 's','ส' : 's','ห' : 'h','ฬ' : 'l','อ' : '?','ฮ' : 'h' }
    stable['Y'] = { 'ก' : 'k', 'ข' : 'k' , 'ค' : 'k', 'ฆ' : 'k', 'ง' : 'N', 'จ' : 't', 'ฉ' : '-', 'ช' : 't', 'ซ' : 't', 'ฌ' : '-','ญ' : 'n','ฎ' : 't','ฏ' : 't','ฐ' : 't','ฑ' : 't','ฒ' : 't','ณ' : 'n','ด' : 't', 'ต' : 't','ถ' : 't','ท' : 't','ธ' : 't','น' : 'n','บ' : 'p','ป' : 'p','ผ' : '-','ฝ' : '-','พ' : 'p','ฟ' : 'p','ภ' : 'p','ม' : 'm','ย' : 'j','ร' : 'n','ฤ' : '-','ล' : 'n','ฦ' : '-','ว' : 'w','ศ' : 't','ษ' : 't','ส' : 't' ,'ห' : '-','ฬ' : 'n','อ' : '-','ฮ' : '-' }

    stable['A'] = stable['X']
    stable['K'] = stable['X']
    stable['C'] = stable['X']
    stable['R'] = stable['X']
    stable['G'] = stable['X']
    stable['E'] = stable['X']
    
    stable['D'] = stable['Y']
    stable['Z'] = stable['Y']
    stable['H'] = stable['Y']
    stable['F'] = stable['Y']

    AK['ก'] = "นฎฐณตถบปมรลวษสพหฬ"
    AK['ข'] = "จนณบมยรลฬดตทษภส"
    AK['ค'] = "กชดตนณปมทฑหบ"
    AK['ฆ'] = "กบรสช"
    AK['จ'] = "ณตนมรลทยกด"
    AK['ฉ'] = "กงนบพรลวทม"
    AK['ช'] = "กญนยรลวฎคทบมอ"
    AK['ซ'] = "ล"
    AK['ญ'] = "กภลญ"
    AK['ณ'] = "กร"
    AK['ด'] = "นรมวบ"
    AK['ต'] = "กตนภยรลฆงถบมวฤท"
    AK['ถ'] = "กนลวศพงมร"
    AK['ท'] = "กชนบมยรลวสหงศ"
    AK['ธ'] = "นภมยรวกพช"
    AK['น'] = "กทธฎภมยรวคขปลวห"
    AK['บ'] = "ดรทพม"
    AK['ป'] = "กฐฏณทนภรวศถฎตปยสหด"
    AK['ผ'] = "งจชดนลสวกคณทยรอ"
    AK['ฝ'] = "ร"
    AK['พ'] = "กญนมยลวสหณจธ"
    AK['ภ'] = "ยรคณมฤวน"
    AK['ม'] = "กณตนยรลหศดธมสฆว"
    AK['ย'] = "กดธภวสบศตมนถช"
    AK['ร'] = "กจณดพภมยวสหชตถนบ"
    AK['ล'] = "กคฆดตอบปพมลวห"
    AK['ว'] = "ชณดนพภรลสมยกจฏตทธปฤศหคธ"
    AK['ศ'] = "จณนบรลวศพดธกตมยส"
    AK['ษ'] = "ณฎบภรนคธม"
    AK['ส'] = "กคงดตถนบปพภมยรลวหอจฟสขทธฤ"
    AK['ห'] = "กงพทนรภญนมยรลวบต"
    AK['อ'] = "กงชณดตธนพภมยรลวสศคบฆจทปห"
    AK['ฑ'] = "มสรนค"
    AK['ฐ'] = "กจ"
    AK['ฏ'] = "ก"
    AK['ฌ'] = "ก"

    EngAbbr = ['เอ','บี','ซี','ดี','อี','เอฟ','จี','เจ','เอช','ไอ','เค','แอล','เอ็ม','เอ็น','โอ','พี','คิว','อาร์','เอส','ที','ยู','วี','เอ็กซ์','เอ็ก','วาย','แซด']

    return(1)

def read_syldict(Filename):
    global PRON
    file1 = open(Filename,'r',encoding='cp874')
    for line in file1:
        if re.match(r'#',line):
            continue
        line = line.rstrip()
        tmp = line.split("\t")
        PRON[tmp[0]].append(tmp[1])
    return(1)

def read_PhSTrigram(File):
    global PhSTrigram
    global FrmSTrigram
    global PhSBigram
    global FrmSBigram
    global PhSUnigram
    global FrmSUnigram
    global AbsUnigram
    global AbsFrmSUnigram
    
    PhSTrigram = defaultdict(float)
    FrmSTrigram = defaultdict(float)
    PhSBigram = defaultdict(float)
    FrmSBigram = defaultdict(float)
    PhSUnigram = defaultdict(float)
    FrmSUnigram = defaultdict(float)
    AbsUnigram = defaultdict(float)
    AbsFrmSUnigram = defaultdict(float)
    
    IFile = open(File,'r',encoding='cp874')
    for line in IFile.readlines():
        line = line.rstrip()
        line = re.sub(r"<w>","|",line)
        (x, ct) = line.split('\t')
        (fs,p) = x.split('/')
        (x1,x2,x3) = fs.split('-')
        PhSTrigram[(x1,x2,x3,p)] += float(ct)
        FrmSTrigram[(x1,x2,x3)] += float(ct)
        PhSBigram[(x1,x2,p)] += float(ct)
        FrmSBigram[(x1,x2)] += float(ct)
        PhSBigram[(x2,x3,p)] += float(ct)
        FrmSBigram[(x2,x3)] += float(ct)
        PhSUnigram[(x2,p)] += float(ct)
        FrmSUnigram[x2] += float(ct)
        abs_x2 = re.sub(r"[่้๊๋]","",x2)
        abs_x2 = re.sub(r"[ก-ฮ]","C",abs_x2)
        abs_p = re.sub(r"[0-9]","",p)
        abs_p = re.sub(r"[^aeio@OuxU]","C",abs_p)
        AbsUnigram[(abs_x2,abs_p)] += float(ct)
        AbsFrmSUnigram[abs_x2] += float(ct)
    IFile.close()

def read_stat(Filename):
    global TriCount
    global BiCount
    global Count
    global BiType
    global Type
    global TotalWord
    global TotalLex

    TriCount = defaultdict(int)
    BiCount = defaultdict(int)
    BiType = defaultdict(int)
    Count = defaultdict(int)
    Type = defaultdict(int)
    
    TotalWord = 0
    TotalLex = 0
    TriCount.clear()
    BiCount.clear()
    Count.clear()
    BiType.clear()
    Type.clear()

    fileObject = open(Filename,'rb')  
    TriCount = pickle.load(fileObject)
    for (X,Y,Z) in TriCount:
        BiType[(X,Y)] += 1
        BiCount[(X,Y)] += TriCount[(X,Y,Z)]
        Count[Y] += TriCount[(X,Y,Z)]

    for (X,Y) in BiCount:
        Type[X] += 1
        
    for X in Count:
        TotalLex += 1
        TotalWord += Count[X]
        
    return(1)

def preprocess(input):
    global SegSep
    global SSegSep

    input = re.sub(r" +ๆ",r"ๆ",input)
    input = re.sub(r"ๆ([^ ])",r"ๆ"+SegSep+r"\1",input)

    NORMALIZE_DICT = [
        ('\u0E40\u0E40', '\u0E41'),
        ('\u0E4D\u0E32', '\u0E33'),
        ('\u0E24\u0E32', '\u0E24\u0E45'),
        ('\u0E26\u0E32', '\u0E26\u0E45'),
    ]
    for k, v in NORMALIZE_DICT:
        input = input.replace(k, v)

    pattern = re.compile(r'([ก-ฮเแาำะไใโฯๆ][\ุ\ู\ึ\ั\ี\๊\้\็\่\๋\ิ\ื\์]*) +([ก-ฮเแาำะไใโฯๆ][\ุ\ู\ึ\ั\ี\๊\้\็\่\๋\ิ\ื\์]*) +|([ก-ฮเแาำะไใโฯๆ][\ุ\ู\ึ\ั\ี\๊\้\็\่\๋\ิ\ื\์]*) +([ก-ฮเแาำะไใโฯๆ][\ุ\ู\ึ\ั\ี\๊\้\็\่\๋\ิ\ื\์]*)$')

    input = re.sub(pattern, r"\1\2", input)

    input = re.sub(r"([^\s\t\u00A0]{3,})[\s\t\u00A0]+([^\s\t\u00A0]+?)",r"\1"+SegSep+r"\2",input)
    input = re.sub(r"([^\s\t\u00A0]+)[\s\t\u00A0]+([0-9]+)",r"\1"+SegSep+r"\2",input)
    input = re.sub(r"([^\s\t\u00A0]+?)[\s\t\u00A0]+([^\s\t\u00A0]{3,})",r"\1"+SegSep+r"\2",input)

    input = re.sub(r"([ก-์][ฯๆ])",r"\1"+SSegSep,input)
    input = re.sub(r"([\u0E01-\u0E5B]+\.?)([^\.\u0E01-\u0E5B\u001F]+)",r"\1"+SSegSep+r"\2",input)
    input = re.sub(r"([^\.\u0E01-\u0E5B\u001F]+)([\u0E01-\u0E5B]+)",r"\1"+SSegSep+r"\2",input)
    input = re.sub(r"(<.+?>)",SSegSep+r"\1",input)
    input = re.sub(r"([0-9a-zA-Z\.\-]{2,})([\u0E01-\u0E5B]+)",r"\1"+SSegSep+r"\2",input)
    input = re.sub(r"(\.\.\.+)",r""+SSegSep+r"\1"+SSegSep,input) 

    return(input)

def initial():
    global SylSep
    global WordSep
    global SegSep
    global SSegSep
    global TDICT
    global PRON
    global CProb
    global SYLVAR

    PRON = defaultdict(list)
    SYLVAR = defaultdict(list)
  
    TDICT = {}
    CProb = defaultdict(float)
    
    SylSep = chr(126)  
    WordSep = chr(124) 
    SSegSep = chr(30)
    SegSep = chr(31)

    path = os.path.abspath(__file__)
    ATA_PATH = os.path.dirname(path)

    read_sylpattern(ATA_PATH + '/data/sylrule.lts')
    read_syldict(ATA_PATH +  '/data/thaisyl.dict')
    read_stat(ATA_PATH + '/data/sylseg.3g')
    read_thdict(ATA_PATH +  '/data/thdict')
    read_PhSTrigram(ATA_PATH +  '/data/PhSTrigram.sts')

    return(1)

initial()

# G2P

SHORT_VOWELS = "aivueyoxz"
LONG_VOWELS =  "AIVUEYOXZ"
DIPHTHONGS = "JWR"
VOWELS = SHORT_VOWELS + LONG_VOWELS + DIPHTHONGS
CLUSTERS = ["br","bl","pr","pl","Pr","Pl","fr","fl","dr","tr","Tr","kr","kl","kw","Kr","Kl","Kw"]
ONSETS = ["b","p","P","m","f","d","t","T","n","s","r","l","c","C","k","K","N","w","j","h","?"]
CODAS = ["p","m","f","t","d","n","s","l","c","k","N","w","j","?","-"]

abs_dir = os.path.dirname(__file__)
with open(abs_dir + '/data/thai2phone.csv', encoding="utf-8") as f:
    THAI2PHONE_DICT = dict(csv.reader(f))
    THAI2PHONE_DICT = {k:v for k,v in THAI2PHONE_DICT.items() if v != ''}

def validate(phone):
    syls = phone.split()
    for syl in syls:
        try:
            tone = syl[-1]
            coda = syl[-2]
            vowel = syl[-3]
            onset = syl[:-3]
        except:
            return False 
        if tone in '12345' and coda in CODAS and vowel in VOWELS and onset in CLUSTERS+ONSETS:
            continue
        else:
            return False
    return True

def clean(text:str):
    text = re.sub(r'[\n\s]+', ' ', text)
    text = re.sub(r'https?://[^\s]+((?=\s)|(?=$))', '', text)
    text = re.sub(r'\((.+?)\)', r'( \1 )', text)
    text = re.sub(r'\"(.+?)\"', r'" \1 "', text)
    text = re.sub(r'[“”„]', '"', text)
    text = re.sub(r'[‘’`]', "'", text)
    text = re.sub(r'[ \u00a0\xa0\u3000\u2002-\u200a\t]+', ' ', text)
    text = re.sub(r'[\r\u200b\ufeff]+', '', text)
    return text.strip()

def get_phone_word(thaiword:str):
    return THAI2PHONE_DICT.get(thaiword, None)

def get_phone_word_tltk(thaiword:str):
    decoded_syls = []
    result = tltkg2p(thaiword)
    tokens = re.findall(r'<tr/>(\S+?)\|(?:<s/>|\s)', result)
    for token in tokens:
        for syl in re.split(r"[|^~\']", token):
            syl = syl.replace('\\', '')
            if not syl or not syl[-1].isdigit():
                tone = "1"
            else:
                tone = str(int(syl[-1]) + 1)
            if int(tone) > 5:
                tone = str(int(tone)-5)
            syl = syl[:-1] + tone
            syl = re.sub(r'iia(?=\d)', 'J-', syl)
            syl = re.sub(r'iia', 'J', syl)
            syl = re.sub(r'ia', 'J-', syl)
            syl = re.sub(r'UUa(?=\d)', 'W-', syl)
            syl = re.sub(r'UUa', 'W', syl)
            syl = re.sub(r'Ua', 'W-', syl)
            syl = re.sub(r'uua(?=\d)', 'R-', syl)
            syl = re.sub(r'uua', 'R', syl)
            syl = re.sub(r'ua', 'R-', syl)
            syl = re.sub(r'aa(?=\d)', 'A-', syl)
            syl = re.sub(r'aa', 'A', syl)
            syl = re.sub(r'a(?=\d)', 'a-', syl)
            syl = re.sub(r'ii(?=\d)', 'I-', syl)
            syl = re.sub(r'ii', 'I', syl)
            syl = re.sub(r'i(?=\d)', 'i-', syl)
            syl = re.sub(r'UU(?=\d)', 'V-', syl)
            syl = re.sub(r'UU', 'V', syl)
            syl = re.sub(r'U(?=\d)', 'v-', syl)
            syl = re.sub(r'U', 'v', syl)
            syl = re.sub(r'uu(?=\d)', 'U-', syl)
            syl = re.sub(r'uu', 'U', syl)
            syl = re.sub(r'u(?=\d)', 'u-', syl)
            syl = re.sub(r'xx(?=\d)', 'Y-', syl)
            syl = re.sub(r'xx', 'Y', syl)
            syl = re.sub(r'x(?=\d)', 'y-', syl)
            syl = re.sub(r'x', 'y', syl)
            syl = re.sub(r'ee(?=\d)', 'E-', syl)
            syl = re.sub(r'ee', 'E', syl)
            syl = re.sub(r'e(?=\d)', 'e-', syl)
            syl = re.sub(r'OO(?=\d)', 'X-', syl)
            syl = re.sub(r'OO', 'X', syl)
            syl = re.sub(r'O(?=\d)', 'x-', syl)
            syl = re.sub(r'O', 'x', syl)
            syl = re.sub(r'oo(?=\d)', 'O-', syl)
            syl = re.sub(r'oo', 'O', syl)
            syl = re.sub(r'o(?=\d)', 'o-', syl)
            syl = re.sub(r'@@(?=\d)', 'Z-', syl)
            syl = re.sub(r'@@', 'Z', syl)
            syl = re.sub(r'@(?=\d)', 'z-', syl)
            syl = re.sub(r'@', 'z', syl)
            syl = re.sub(r'th', 'T', syl)
            syl = re.sub(r'kh', 'K', syl)
            syl = re.sub(r'ph', 'P', syl)
            syl = re.sub(r'ch', 'C', syl)
            decoded_syls.append(syl)
    return ' '.join(decoded_syls)

def decode(phone, transcription='haas', keep_space=True):
    if type(phone) == str:
        syls = phone.split()
    elif type(phone) == list:
        syls = phone
    else:
        raise TypeError
    decoded_syls = []
    for syl in syls:
        if not validate(syl):
            decoded_syls.append(syl)
            continue
        tone = syl[-1]
        coda = syl[-2].replace('ʔ','-')
        vowel = syl[-3]
        onset = syl[:-3]
        if transcription.lower() == 'ipa':
            decoded_syls.append(''.join([PHONE2IPA[c] for c in onset]) + PHONE2IPA[vowel+tone] + PHONE2IPA[coda])
        elif transcription.lower() == 'haas':
            decoded_syls.append(''.join([PHONE2HAAS[c] for c in onset]) + PHONE2HAAS[vowel+tone] + PHONE2HAAS[coda])
        elif transcription.lower() == 'rtgs':
            decoded_syls.append(''.join([PHONE2RTGS[c] for c in onset]) + PHONE2RTGS[vowel+tone] + PHONE2RTGS_CODA[coda])
    if keep_space == False:
        decoded_syls = ''.join(decoded_syls)
    elif type(keep_space) == str:
        decoded_syls = keep_space.join(decoded_syls)
    else:
        decoded_syls = ''.join(decoded_syls)
    return decoded_syls

def g2p(sentence, transcription='haas', return_tokens=False, decoded=True):
    if type(sentence) == str:
        sentence = clean(sentence)
    elif type(sentence) == list and type(sentence[0]) == str:
        tokens = sentence
    token_phone_list = []
    for i, token in enumerate(tokens):
        if token == 'น.' and i > 0 and\
        (token_phone_list[-1][1].endswith('nA-1 li-4 kA-1') or token_phone_list[-1][1].endswith('nA-1 TI-1')):
            token_phone_list[-1][0] += ' น.'
            continue
        elif token == 'ๆ' and i > 0:
            token_phone_list[-1][0] += ' ๆ'
            token_phone_list[-1][1] += ' ' + token_phone_list[-1][1]
            continue
        elif token in THAI2PHONE_DICT:
            phone = get_phone_word(token)
        elif re.match('[ก-ฮ]$', token):
            continue
        elif re.match(r'[ก-๙][ก-๙\-\.]*$', token):
            phone = get_phone_word_tltk(token)
        else:
            phone = token
        token_phone_list.append([token, phone])
    if decoded:
        token_phone_list = [[t, decode(p, transcription)] for t, p in token_phone_list]
    if return_tokens:
        return token_phone_list
    else:
        return ' '.join([phone for _, phone in token_phone_list])

PHONE2IPA = {
    'a1':'a' ,'a2':'à' ,'a3':'â' ,'a4':'á' ,'a5':'ǎ' ,
    'A1':'aː','A2':'àː','A3':'âː','A4':'áː','A5':'ǎː',
    'i1':'i' ,'i2':'ì' ,'i3':'î' ,'i4':'í' ,'i5':'ǐ' ,
    'I1':'iː','I2':'ìː','I3':'îː','I4':'íː','I5':'ǐː',
    'u1':'u' ,'u2':'ù' ,'u3':'û' ,'u4':'ú' ,'u5':'ǔ' ,
    'U1':'uː','U2':'ùː','U3':'ûː','U4':'úː','U5':'ǔː',
    'v1':'ɯ' ,'v2':'ɯ̀' ,'v3':'ɯ̂' ,'v4':'ɯ́' ,'v5':'ɯ̌' ,
    'V1':'ɯː','V2':'ɯ̀ː','V3':'ɯ̂ː','V4':'ɯ́ː','V5':'ɯ̌ː',
    'e1':'e' ,'e2':'è' ,'e3':'ê' ,'e4':'é' ,'e5':'ě' ,
    'E1':'eː','E2':'èː','E3':'êː','E4':'éː','E5':'ěː',
    'y1':'ɛ' ,'y2':'ɛ̀' ,'y3':'ɛ̂' ,'y4':'ɛ́' ,'y5':'ɛ̌' ,
    'Y1':'ɛː','Y2':'ɛ̀ː','Y3':'ɛ̂ː','Y4':'ɛ́ː','Y5':'ɛ̌ː',
    'o1':'o' ,'o2':'ò' ,'o3':'ô' ,'o4':'ó' ,'o5':'ǒ' ,
    'O1':'oː','O2':'òː','O3':'ôː','O4':'óː','O5':'ǒː',
    'x1':'ɔ' ,'x2':'ɔ̀' ,'x3':'ɔ̂' ,'x4':'ɔ́' ,'x5':'ɔ̌' ,
    'X1':'ɔː','X2':'ɔ̀ː','X3':'ɔ̂ː','X4':'ɔ́ː','X5':'ɔ̌ː',
    'z1':'ə' ,'z2':'ə̀' ,'z3':'ə̂' ,'z4':'ə́' ,'z5':'ə̌' ,
    'Z1':'əː','Z2':'ə̀ː','Z3':'ə̂ː','Z4':'ə́ː','Z5':'ə̌ː',
    'J1':'iə','J2':'ìə','J3':'îə','J4':'íə','J5':'ǐə',
    'W1':'ɯə','W2':'ɯ̀ə','W3':'ɯ̂ə','W4':'ɯ́ə','W5':'ɯ̌ə',
    'R1':'uə','R2':'ùə','R3':'ûə','R4':'úə','R5':'ǔə',
    'b' :'b' ,'p' :'p' ,'P' :'pʰ','m' :'m' ,'f' :'f' ,
    'd' :'d' ,'t' :'t' ,'T' :'tʰ','n' :'n' ,'s' :'s' ,
    'r' :'r' ,'l' :'l' ,'c' :'tɕ','C' :'tɕʰ',
    'k' :'k' ,'K' :'kʰ','N' :'ŋ' ,
    'w' :'w' ,'j' :'j' ,'h' :'h' ,'?' :'ʔ',
    '.' :'.' ,'-' :''
}

PHONE2HAAS = {
    'a1':'a' ,'a2':'à' ,'a3':'â' ,'a4':'á' ,'a5':'ǎ' ,
    'A1':'aa','A2':'àa','A3':'âa','A4':'áa','A5':'ǎa',
    'i1':'i' ,'i2':'ì' ,'i3':'î' ,'i4':'í' ,'i5':'ǐ' ,
    'I1':'ii','I2':'ìi','I3':'îi','I4':'íi','I5':'ǐi',
    'u1':'u' ,'u2':'ù' ,'u3':'û' ,'u4':'ú' ,'u5':'ǔ' ,
    'U1':'uu','U2':'ùu','U3':'ûu','U4':'úu','U5':'ǔu',
    'v1':'ɯ' ,'v2':'ɯ̀' ,'v3':'ɯ̂' ,'v4':'ɯ́' ,'v5':'ɯ̌' ,
    'V1':'ɯɯ','V2':'ɯ̀ɯ','V3':'ɯ̂ɯ','V4':'ɯ́ɯ','V5':'ɯ̌ɯ',
    'e1':'e' ,'e2':'è' ,'e3':'ê' ,'e4':'é' ,'e5':'ě' ,
    'E1':'ee','E2':'èe','E3':'êe','E4':'ée','E5':'ěe',
    'y1':'ɛ' ,'y2':'ɛ̀' ,'y3':'ɛ̂' ,'y4':'ɛ́' ,'y5':'ɛ̌' ,
    'Y1':'ɛɛ','Y2':'ɛ̀ɛ','Y3':'ɛ̂ɛ','Y4':'ɛ́ɛ','Y5':'ɛ̌ɛ',
    'o1':'o' ,'o2':'ò' ,'o3':'ô' ,'o4':'ó' ,'o5':'ǒ' ,
    'O1':'oo','O2':'òo','O3':'ôo','O4':'óo','O5':'ǒo',
    'x1':'ɔ' ,'x2':'ɔ̀' ,'x3':'ɔ̂' ,'x4':'ɔ́' ,'x5':'ɔ̌' ,
    'X1':'ɔɔ','X2':'ɔ̀ɔ','X3':'ɔ̂ɔ','X4':'ɔ́ɔ','X5':'ɔ̌ɔ',
    'z1':'ə' ,'z2':'ə̀' ,'z3':'ə̂' ,'z4':'ə́' ,'z5':'ə̌' ,
    'Z1':'əə','Z2':'ə̀ə','Z3':'ə̂ə','Z4':'ə́ə','Z5':'ə̌ə',
    'J1':'ia','J2':'ìa','J3':'îa','J4':'ía','J5':'ǐa',
    'W1':'ɯa','W2':'ɯ̀a','W3':'ɯ̂a','W4':'ɯ́a','W5':'ɯ̌a',
    'R1':'ua','R2':'ùa','R3':'ûa','R4':'úa','R5':'ǔa',
    'b' :'b' ,'p' :'p' ,'P' :'ph','m' :'m' ,'f' :'f' ,
    'd' :'d' ,'t' :'t' ,'T' :'th','n' :'n' ,'s' :'s' ,
    'r' :'r' ,'l' :'l' ,'c' :'c' ,'C' :'ch',
    'k' :'k' ,'K' :'kh','N' :'ŋ' ,
    'w' :'w' ,'j' :'y' ,'h' :'h' ,'?' :'ʔ' ,
    '.':'.',   '-':''
}

PHONE2RTGS = {
    'a1':'a' ,'a2':'a' ,'a3':'a' ,'a4':'a' ,'a5':'a',
    'A1':'a' ,'A2':'a' ,'A3':'a' ,'A4':'a' ,'A5':'a',
    'i1':'i' ,'i2':'i' ,'i3':'i' ,'i4':'i' ,'i5':'i',
    'I1':'i' ,'I2':'i' ,'I3':'i' ,'I4':'i' ,'I5':'i',
    'u1':'u' ,'u2':'u' ,'u3':'u' ,'u4':'u' ,'u5':'u',
    'U1':'u' ,'U2':'u' ,'U3':'u' ,'U4':'u' ,'U5':'u',
    'v1':'ue','v2':'ue','v3':'ue','v4':'ue','v5':'ue',
    'V1':'ue','V2':'ue','V3':'ue','V4':'ue','V5':'ue',
    'e1':'e' ,'e2':'e' ,'e3':'e' ,'e4':'e' ,'e5':'e',
    'E1':'e' ,'E2':'e' ,'E3':'e' ,'E4':'e' ,'E5':'e',
    'y1':'ae','y2':'ae','y3':'ae','y4':'ae','y5':'ae',
    'Y1':'ae','Y2':'ae','Y3':'ae','Y4':'ae','Y5':'ae',
    'o1':'o' ,'o2':'o' ,'o3':'o' ,'o4':'o' ,'o5':'o',
    'O1':'o' ,'O2':'o' ,'O3':'o' ,'O4':'o' ,'O5':'o',
    'x1':'o' ,'x2':'o' ,'x3':'o' ,'x4':'o' ,'x5':'o',
    'X1':'o' ,'X2':'o' ,'X3':'o' ,'X4':'o' ,'X5':'o',
    'z1':'oe','z2':'oe','z3':'oe','z4':'oe','z5':'oe',
    'Z1':'oe','Z2':'oe','Z3':'oe','Z4':'oe','Z5':'oe',
    'J1':'ia','J2':'ia','J3':'ia','J4':'ia','J5':'ia',
    'W1':'uea','W2':'uea','W3':'uea','W4':'uea','W5':'uea',
    'R1':'ua','R2':'ua','R3':'ua','R4':'ua','R5':'ua',
    'b' :'b' ,'p' :'p' ,'P' :'ph','m' :'m' ,'f' :'f' ,
    'd' :'d' ,'t' :'t' ,'T' :'th','n' :'n' ,'s' :'s' ,
    'r' :'r' ,'l' :'l' ,'c' :'ch' ,'C' :'ch',
    'k' :'k' ,'K' :'kh','N' :'ng' ,
    'w' :'w' ,'j' :'y' ,'h' :'h' ,'?' :'' ,
    '.':'.',   '-':''
}

PHONE2RTGS_CODA = {
    'b' :'b' ,'p' :'p' ,'P' :'ph','m' :'m' ,'f' :'f' ,
    'd' :'d' ,'t' :'t' ,'T' :'th','n' :'n' ,'s' :'s' ,
    'r' :'r' ,'l' :'l' ,'c' :'ch' ,'C' :'ch',
    'k' :'k' ,'K' :'kh','N' :'ng' ,
    'w' :'o' ,'j' :'i' ,'h' :'h' ,'?' :'' ,
    '.':'.',   '-':''
}

if __name__ == "__main__":
    text = ["สวัสดีครับ","นี่คือเสียง","พูดภาษาไทย"]
    ipa = g2p(text,transcription="ipa")
    print(ipa)