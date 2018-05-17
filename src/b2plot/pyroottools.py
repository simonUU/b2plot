import pandas as pd
import numpy as np
import ROOT as root
import root_pandas
import os
import math
from array import array
from scipy.stats import binned_statistic
from array import array

# load rootlogon
#root.gROOT.Macro( os.path.expanduser( '/nfs/dust/belle2/user/ferber/git/jupyter_nb/libs/rootlogon.C' ) )
#root.gROOT.Macro( os.path.expanduser( 'rootlogon.C' ) )

# default canvas sizes
kCanvasWidth = 700;
kCanvasHeight1 = 500;
kCanvasHeight2 = 600;

# default palette
root.gStyle.SetPalette(112) #kViridis = 112

# ------------------------------------------------------------
def MakeAndFillTGraph(df, varx, vary, color = root.kBlack, markersize=1.0, markerstyle=20, title='graph'):
    x = np.array(df[varx],dtype='float64')
    y = np.array(df[vary],dtype='float64')
    n = len(x)

    g = root.TGraph(n, x, y)
    g.SetMarkerColor(color)
    g.SetLineColor(color)
    g.SetMarkerStyle(markerstyle)
    g.SetMarkerSize(markersize)
    g.SetTitle(title)

    return g


# ------------------------------------------------------------
def ScaleTGraphXY(tg, fx=1, fy=1):
    n = tg.GetN();
    tg_new = root.TGraph(n);

    for i in range(n):
        x, y = root.Double(0), root.Double(0)
        tg.GetPoint(i, x, y)
        tg_new.SetPoint(i, x*fx, y*fy)

    return tg_new

# ------------------------------------------------------------
# add a column that contains a categorical variables (bincount)
def AddBinColumn(df, xmin, xmax, xbins, var):
    if xbins <1:
        print('Invalid number of bins', xbins)
        return None

    binedges = np.linspace(xmin, xmax, num=xbins+1)
    binlabels = np.linspace(int(1), int(xbins), num=xbins)

    bincol = '%s_%s' % (var, 'bin')
    #bincolcount = '%s_%s' % (var, 'count')

    df[bincol] = pd.cut(df[var], bins=binedges, labels=binlabels)
    #df[bincolcount] = df.groupby(bincol)[bincol].transform('count')

# ------------------------------------------------------------
def FillImage(hist, df, var, row):
  #print(row, flush=True)
    x=1
    y=0
    for val in range(0, 49):
        if val%7==0:
            x = 1
            y = y+1
        hist.SetBinContent(x, y, hist.GetBinContent(x, y) + float(df[row:row+1]["%s_%i" % (var, val)]))
        x = x+1
    return hist

# ------------------------------------------------------------
def GetUnderOverFlow(hists = [], minperc = 0.001):

    uotexts = []
    for h in hists:
        nbins = h.GetNbinsX()
        integral = h.Integral(0,nbins+1);

        underflow = h.GetBinContent(0)/integral*100
        overflow  = h.GetBinContent(nbins+1)/integral*100

        # underflow
        if underflow>minperc:
            uflow = '%.3f%%' % underflow
        elif underflow<=minperc:
            uflow = '<%.3f%%' % minperc
        if underflow == 0.:
            uflow = 'none'

        # overflow
        if overflow>minperc:
            oflow = '%.3f%%' % overflow
        elif overflow<=minperc:
            oflow = '<%.3f%%' % minperc
        if overflow == 0.:
            oflow = 'none'

        uoflow = 'U/O-flow (%s): %s / %s' % (h.GetTitle(), uflow, oflow)

        t = root.TText( 0.935 + len(uotexts)*0.02, 0.25, uoflow );
        t.SetNDC();
        t.SetTextSize(root.gStyle.GetTitleSize("X")*0.6);
        t.SetTextAngle(90);

        uotexts.append(t)

    return uotexts

# ------------------------------------------------------------
def ChangeStyleTObject(tobject, factortitle, factortext):

    factoroffset = 1.0;
    hightitleoffsety = 1.0;
    hightitleoffsetx = 1.0;
    if factortitle>1.3:
        factoroffset = 0.875;
    if factortitle>=1.5:
        factoroffset = 0.95;
    if factortitle>=1.75:
        factoroffset = 0.95
        hightitleoffsety = 1.75
        hightitleoffsetx = 1.15
        factortitle = 1.25

    if tobject.InheritsFrom(root.TH1.Class()):
        tobject.GetXaxis().SetTitleSize(root.gStyle.GetTitleSize("X")*factortitle)
        tobject.GetXaxis().SetLabelSize(root.gStyle.GetLabelSize("X")*factortitle)
        tobject.GetYaxis().SetTitleSize(root.gStyle.GetTitleSize("Y")*factortitle)
        tobject.GetYaxis().SetLabelSize(root.gStyle.GetLabelSize("Y")*factortitle)
        tobject.GetZaxis().SetTitleSize(root.gStyle.GetTitleSize("Z")*factortitle)
        tobject.GetZaxis().SetLabelSize(root.gStyle.GetLabelSize("Z")*factortitle)

    if tobject.InheritsFrom(root.THStack.Class()):
        tobject.GetXaxis().SetTitleSize(root.gStyle.GetTitleSize("X")*factortitle)
        tobject.GetXaxis().SetLabelSize(root.gStyle.GetLabelSize("X")*factortitle)
        tobject.GetYaxis().SetTitleSize(root.gStyle.GetTitleSize("Y")*factortitle)
        tobject.GetYaxis().SetLabelSize(root.gStyle.GetLabelSize("Y")*factortitle)

    if tobject.InheritsFrom(root.TGraph.Class()):
        tobject.GetXaxis().SetTitleSize(root.gStyle.GetTitleSize("X")*factortitle)
        tobject.GetXaxis().SetLabelSize(root.gStyle.GetLabelSize("X")*factortitle)
        tobject.GetYaxis().SetTitleSize(root.gStyle.GetTitleSize("Y")*factortitle)
        tobject.GetYaxis().SetLabelSize(root.gStyle.GetLabelSize("Y")*factortitle)

    if tobject.InheritsFrom(root.TLegend.Class()):
        tobject.SetTextSize(root.gStyle.GetTitleSize("X")*0.8*factortext)

    #if tobject.InheritsFrom(root.TPaveText.Class()):
     #   tobject.SetTextSize(root.GetTextSize("X")*0.8*factortext)

    return tobject

# ------------------------------------------------------------
def ChangeStyle(c, factortitle = 1.25, factortext = 1.0):
    for tobject in c.GetListOfPrimitives():
        if tobject.InheritsFrom(root.TPad.Class()):
            ChangeStyle(tobject, factortitle, factortext)
        else:
            o = ChangeStyleTObject(tobject, factortitle, factortext)
    c.Update()
    return c

# ------------------------------------------------------------
def SavePdf(c, name = '', savedir = ''):
    SaveAs(c=c, name=name, savedir=savedir, filetype='pdf')

#------------------------------------------------------------
def SavePng(c, name = '', savedir = ''):
    SaveAs(c=c, name=name, savedir=savedir, filetype='png')

# ------------------------------------------------------------
def SaveAs(c, name = '', savedir = '', filetype='pdf'):
    filetypes = ('pdf', 'png')

    if filetype not in filetypes:
        print('SaveAs: Wrong filetype', filetype)
        return None

    if savedir == '':
        savedir = '.'

    savedir = os.path.join(savedir,'')

    if not os.path.exists(savedir):
        os.makedirs(savedir)

    if name == '':
        name = c.GetName()

    name = '%s%s.%s' % (savedir, name, filetype)
    ChangeStyle(c)
    c.Print(name)

# ------------------------------------------------------------
def MakeTLegend(xpos=0.225, ypos=0.9, title='', tobjects=[], options=[]):

    entries = len(tobjects)
    if title != '':
        entries = entries + 1

    xl = xpos
    width = 0.25
    xh = xl + width
    yh = ypos
    yl = yh - entries*0.05

    leg = root.TLegend(xl, yl, xh, yh)
    leg.SetTextAlign(12)
    leg.SetTextSize(root.gStyle.GetTitleSize("X")*0.8)
    leg.SetTextFont(43)
    leg.SetLineColor(0)
    leg.SetFillColor(root.kWhite)
    leg.SetLineWidth(0)
    leg.SetFillStyle(0)
    leg.SetNColumns(1)
    leg.SetMargin(0.3)
    leg.SetBorderSize(0)

    if title!= '':
        leg.SetHeader(title)

    for i, tobject in enumerate(tobjects):
        leg.AddEntry(tobject, tobject.GetTitle(), options[i])

    return leg


# ------------------------------------------------------------
def MakeFillAndDrawTH1Ds(dfs=[], canvasname ='mycanvas', var=None, titles=[], colors=[],
                         nbins=101, low=-0.005, high=1.005, xtitle='x', ytitle='y', text = '',
                         show=False, save=True, savedir='saveplots', legend=True, legxpos=0.225,
                         legypos=0.9, uoflow=True, unity=False, invert=False, filled=False, grid=True,
                         setlog=False, extratext=[], extratextx = 0.65, extratexty=0.9, normalize=False):

    if show == False:
        root.gROOT.SetBatch(True)

    if len(colors) < len(dfs):
        print('Need more colors defined to plot all dataframes!')
        return None

    if len(titles) < len(dfs):
        print('Need more titles defined to plot all dataframes!')
        return None

    hists = []
    legoptions = []
    mycanvas = MakeTCanvas(canvasname, canvasname, 0)
    max = 0
    stack = root.THStack()

    # get maximal range of all dataframes
    values_min = float('Inf')
    values_max = -float('Inf')
    if unity:
        for i, df in enumerate(dfs):
            values = df[var].as_matrix()
            vmin = values.min() #minimum entry
            vmax = values.max() #maximum entry
            if vmin < values_min:
                values_min = vmin
            if vmax > values_max:
                values_max = vmax

    for i, df in enumerate(dfs):
        title = 'hist_%i'%i
        if titles[i]:
             title = titles[i]

        hist = MakeTH1D( 'hist_%i'%i, title, nbins, low, high, xtitle, ytitle, linecolor=colors[i])

        # scale histograms to 0..1
        if unity:

            hist = FillHist1DUnityInvert(hist, dfs[i], var, invert, minmax=(values_min, values_max))

            if normalize:
                norm = hist.Integral(0, hist.GetNbinsX()+1)
                hist.Scale(1./norm)
                print(norm)

            stack.Add(hist, 'HIST')
            legoptions.append('F')
        else:
            FillHist1D(hist, dfs[i], var)

            if normalize:
                norm = hist.Integral(0, hist.GetNbinsX()+1)
                hist.Scale(1./norm)
                print(norm)

            if filled:
                stack.Add(hist, 'HIST')
                legoptions.append('F')
            else:
                stack.Add(hist, 'E0')
                legoptions.append('LP')


        hists.append(hist)

    # use a stack to automatically get the right y-axis range incl. errorbars
    stack.Draw('nostack')
    stack.GetXaxis().SetTitle(hists[0].GetXaxis().GetTitle())
    stack.GetYaxis().SetTitle(hists[0].GetYaxis().GetTitle())
    stack.GetHistogram().Draw('0')

    if unity:
        stack.Draw('nostack HIST same')
    else:
        if filled:
            stack.Draw('nostack HIST same')
        else:
            stack.Draw('nostack E0 same')

    root.gPad.RedrawAxis()

    # plot legend
    leg = 'None'
    if legend:
        leg = MakeTLegend(xpos=legxpos, ypos=legypos, title='', tobjects=hists, options=legoptions)
        leg.Draw()

    # plot under and overflow
    uotexts = []
    if uoflow:
        uotexts = GetUnderOverFlow(hists=hists)
        for uotext in uotexts:
            uotext.Draw()
    if grid:
        root.gPad.SetGridx()
        root.gPad.SetGridy()

    if text != '':
        tinfo = MakeTPText(text=text, x1 = 0.3)
        tinfo.Draw()

    if len(extratext):
        extralist = []
        for i, text in enumerate(extratext):
            t = MakeTPText(text=text, x1 = extratextx, y1 = extratexty - (i+1)*0.05, y2 = extratexty - (i)*0.05)
            t.Draw()
            extralist.append(t)

    # logscale
    if setlog:
        root.gPad.SetLogy()

    # display
    if show:
        mycanvas.Draw()

    # save to pdf
    if save:
        SavePdf(mycanvas, savedir=savedir)

    # return things to make sure they stay alive
    return mycanvas, hists, stack, uotexts, leg

# ------------------------------------------------------------
def MakeFillAndDrawTH2Ds(df,
                         canvasname ='mycanvas',
                         varx = None,
                         vary = None,
                         title ='',
                         nbinsx = 101,
                         lowx = -0.005,
                         highx = 1.005,
                         nbinsy = 101,
                         lowy = -0.005,
                         highy = 1.005,
                         titlex = 'x',
                         titley = 'y',
                         titlez = 'z',
                         text = '',
                         logz = True,
                         show = False,
                         save = True,
                         savedir = 'saveplots',
                         grid = True):

    if show == False:
        root.gROOT.SetBatch(True)

    # make TCanvas
    mycanvas = MakeTCanvas(canvasname, canvasname, 2)

    # make TH2D
    hist = MakeTH2D( 'hist2d', title, nbinsx, lowx, highx, nbinsy, lowy, highy, titlex, titley, titlez)

    # fill TH2D
    FillHist2D(hist, df, varx, vary, scalex=1, scaley=1.0)

    # draw
    hist.Draw('COLZ')

    if grid:
        root.gPad.SetGridx()
        root.gPad.SetGridy()

    if text != '':
        tinfo = MakeTPText(text=text, x1 = 0.3)
        tinfo.Draw()

    # log
    if logz:
        root.gPad.SetLogz()

    # display
    if show:
        mycanvas.Draw()

    # save to pdf
    if save:
        SavePdf(mycanvas, savedir=savedir)

    # return things to make sure they stay alive
    return mycanvas, hist

# ------------------------------------------------------------
def FillHist1DUnityInvert(hist, df, var, invert=False, minmax=(-float('Inf'), float('Inf'))):
    hist.Reset()

    xtitle= ''
    if invert:
        xtitle = '%s_{mirrored, scaled (0..1)}' % (hist.GetXaxis().GetTitle())
    else:
        xtitle = '%s_{scaled (0..1)}' % (hist.GetXaxis().GetTitle())

    hist.GetXaxis().SetTitle(xtitle)

    values = df[var].as_matrix()
    for x in np.nditer(values):
        scaled = (x-minmax[0])/(minmax[1]-minmax[0])

        if invert:
            scaled = 1-scaled

        hist.Fill(scaled)

    return hist

# ------------------------------------------------------------
def FillHist1D(hist, df, var, xscale=1.0):

    n = hist.GetNbinsX()
    x0 = hist.GetBinLowEdge(1)
    x1 = hist.GetBinLowEdge(n) + hist.GetBinWidth(n)

    if len(df)>0:
        out = binned_statistic(np.array(df[var],dtype='float64'), np.array(df[var],dtype='float64'), bins=n, range=(x0, x1), statistic='count')

        if len(out)>0:
            statistic = out[0]

            for v in range(len(statistic)):
                hist.SetBinContent(v+1, statistic[v]*xscale)

            hist.SetBinContent(0, len(df[df[var]<x0])) #underflow
            hist.SetBinContent(n+1, len(df[df[var]>x1])) #overflow


# ------------------------------------------------------------
def FillHist2D(hist, df, var1, var2, scalex=1, scaley=1.0):
    values1 = df[var1].as_matrix()
    values2 = df[var2].as_matrix()
    for x, y in np.c_[values1,values2]:
        hist.Fill(x*scalex, y*scaley)
    return hist

# ------------------------------------------------------------
def SetTitlesTH1(h1, xtit, ytit):
    xax = h1.GetXaxis();
    yax = h1.GetYaxis();
    xax.SetTitle(xtit);
    yax.SetTitle(ytit);

# ------------------------------------------------------------
def SetTitlesTH2(h2, xtit, ytit, ztit):
    xax = h2.GetXaxis();
    yax = h2.GetYaxis();
    zax = h2.GetZaxis();
    xax.SetTitle(xtit);
    yax.SetTitle(ytit);
    zax.SetTitle(ztit);
    zax.SetTitleOffset(1.15); #needed for COLZ drawing

def MakeTPText(text=None, x1=0.25, y1=0.9375, x2=1.0, y2=1.0, fillcolor=0, fillstyle=0, factortext=0.8, textcolor=1, textsize=None, textalign=13):
    tp = root.TPaveText( x1, y1, x2, y2, "NDC" );
    tp.SetBorderSize(0);
    tp.SetMargin(0);
    tp.SetTextAlign(textalign);
    tp.AddText(0.0,0.0,text);
    tp.SetFillColor(fillcolor);
    tp.SetFillStyle(fillstyle);
    tp.SetTextColor(textcolor);
    if textsize==None:
        tp.SetTextSize(root.gStyle.GetTitleSize("X")*factortext);
    else:
        tp.SetTextSize(textsize);

    return tp

# ------------------------------------------------------------
def MakeTH1D( name, title, nbins, low, high, xtitle, ytitle, linecolor=1, linewidth=2, alpha=0.5):
    h1 = root.TH1D(name,title,nbins,low,high);
    SetTitlesTH1(h1, xtitle, ytitle);
    h1.SetMarkerColor(linecolor)
    h1.SetLineColor(linecolor)
    h1.SetFillColorAlpha(linecolor, alpha)
    h1.SetLineWidth(linewidth)
    return h1;

# ------------------------------------------------------------
def MakeTH2D( name, title, nbins, low, high, nbinsy, lowy, highy, xtitle, ytitle, ztitle):
    h2 = root.TH2D(name,title,nbins,low,high,nbinsy,lowy,highy);
    SetTitlesTH2(h2, xtitle, ytitle, ztitle);
    return h2;

# ------------------------------------------------------------
# type:
# 0: TH1 plots
# 1: Ratio plots
# 2: COLZ TH2 plots (with color palette)
# 8: quadratic histogram (e.g. for display of correlation matrices), same width as type=0
# 9: quadratic histogram, same width as type=2
# 93: 3 quadratic next to each other, no margins

def MakeTCanvas(name='c_name', title='c_title', type=0):

    w = kCanvasWidth;
    h = kCanvasHeight1;

    rightmargin  = root.gStyle.GetPadRightMargin();
    leftmargin   = root.gStyle.GetPadLeftMargin();
    bottommargin = root.gStyle.GetPadBottomMargin();
    topmargin    = root.gStyle.GetPadTopMargin();

    colzrightmargin = 0.19;

    # different types here...
    if type == 1:
        h = kCanvasHeight2;
    if type == 2:
        root.gStyle.SetPadRightMargin(colzrightmargin)
    elif type==8:
        weff = w*(1-root.gStyle.GetPadLeftMargin()-root.gStyle.GetPadRightMargin())
        heff = h*(1-root.gStyle.GetPadTopMargin()-root.gStyle.GetPadBottomMargin())
        topm    = h*root.gStyle.GetPadTopMargin()
        bottomm = h*root.gStyle.GetPadBottomMargin()

        h = int(weff+topm+bottomm)
        root.gStyle.SetPadTopMargin(topm/h);
        root.gStyle.SetPadBottomMargin(bottomm/h);
    elif type==9:
        root.gStyle.SetPadRightMargin(colzrightmargin);
        weff = w*(1-root.gStyle.GetPadLeftMargin()-root.gStyle.GetPadRightMargin())
        heff = h*(1-root.gStyle.GetPadTopMargin()-root.gStyle.GetPadBottomMargin())
        topm    = h*root.gStyle.GetPadTopMargin()
        bottomm = h*root.gStyle.GetPadBottomMargin()

        h = int(weff+topm+bottomm)
        root.gStyle.SetPadTopMargin(topm/h);
        root.gStyle.SetPadBottomMargin(bottomm/h);

    elif type==93:
        w = 1000
        h = 300
        topm    = 0
        bottomm = 0

        root.gStyle.SetPadRightMargin(0);
        root.gStyle.SetPadLeftMargin(0);
        root.gStyle.SetPadTopMargin(0);
        root.gStyle.SetPadBottomMargin(0);


    c = root.TCanvas(name, title, w, h)
    c.SetCanvasSize(w, h);
    root.gPad.Update();

    pads = []
    if type==93:
        c.cd()
        pad1 = root.TPad("pad1","pad1",0.01,0.01,0.29,0.89);
        pad1.Draw()
        pads.append(pad1)

        c.cd()
        pad1t = root.TPad("pad1t","pad1t",0.0,0.9,0.3,1.0);
        pad1t.Draw()
        pads.append(pad1t)

        c.cd()
        pad2 = root.TPad("pad2","pad2",0.31,0.0,0.59,0.89);
        pad2.Draw()
        pads.append(pad2)

        c.cd()
        pad2t = root.TPad("pad2t","pad2t",0.3,0.9,0.6,1.0);
        pad2t.Draw()
        pads.append(pad2t)

        c.cd()
        pad3 = root.TPad("pad3","pad3",0.61,0.0,0.89,0.89);
        pad3.Draw()
        pads.append(pad3)

        c.cd()
        pad3t = root.TPad("pad3t","pad3t",0.6,0.9,0.9,1.0);
        pad3t.Draw()
        pads.append(pad3t)

        c.cd()
        pad4t = root.TPad("pad4t","pad4t",0.9,0.0,1.0,1.0);
        pad4t.Draw()
        pads.append(pad4t)




    root.gStyle.SetPadRightMargin(rightmargin);
    root.gStyle.SetPadLeftMargin(leftmargin);
    root.gStyle.SetPadBottomMargin(bottommargin);
    root.gStyle.SetPadTopMargin(topmargin);

    if type==93:
        return c, pads
    return c

def ROCScan(hists, flipped = False, gtitle='', gcolor=root.kBlack, glinestyle=1, save=False, savedir='saveplots', canvasname='mycanvas'):
    g = root.TGraph() #ROC graph
    g_auc = root.TGraph() #ROC graph
    frac = 0 #integral overlaps

    if len(hists)!=2:
        print('ROCScan: Wrong number of histograms (must be two): ', len(hists))
        return None

    integral = (hists[0].Integral(), hists[1].Integral())
    fraction = [0., 0.]
    roc = []

    if flipped==False:
        for i in range(1, hists[0].GetNbinsX()+1):
            fraction[0] = fraction[0] + hists[0].GetBinContent(i)
            fraction[1] = fraction[1] + hists[1].GetBinContent(i)
            roc.append((fraction[1]/integral[1],1-fraction[0]/integral[0]))
    else:
         for i in range(hists[0].GetNbinsX()+1,1,-1):
            fraction[0] = fraction[0] + hists[0].GetBinContent(i)
            fraction[1] = fraction[1] + hists[1].GetBinContent(i)
            roc.append((fraction[1]/integral[1],1-fraction[0]/integral[0]))

    df_roc = pd.DataFrame.from_records(roc, columns=('purity', 'efficiency'))
    df_roc.sort_values('purity')
    for index, row in df_roc.iterrows():
        g.SetPoint(g.GetN(), row['purity'], row['efficiency'])

    g.GetHistogram().GetXaxis().SetTitle("Purity")
    g.GetHistogram().GetYaxis().SetTitle("1-Efficiency")

    g.SetTitle('%s' % (hists[0].GetTitle()))
    if gtitle!='':
        g.SetTitle('%s' % gtitle)

    # get integral differences (0..2)
    for i in range(1, hists[0].GetNbinsX()+1):
        diff = abs(hists[0].GetBinContent(i) - hists[1].GetBinContent(i))
        frac += diff
    frac /=integral[0]

    # set color
    g.SetLineColor(gcolor)
    g.SetLineStyle(glinestyle)

    # get area under curve (0..1)
    g_auc = g.Clone()
    g_auc.SetPoint(g_auc.GetN(), 0.0, 0.0)
    g_auc.SetPoint(g_auc.GetN(), 0.0, 1.0)
    auc = g_auc.Integral()

    # plot and save
    if save:
        c = MakeTCanvas(canvasname, canvasname, 8)
        g.Draw('AC')
        root.gPad.SetGridx()
        root.gPad.SetGridy()

        # add AUC
        tpauc = MakeTPText(text='AUC=%2.3f'%auc, x1=0.7, y1=0.8)
        tpauc.Draw()

        # add title
        tptitle = MakeTPText(text=gtitle)
        tptitle.Draw()

        SavePdf(c, savedir=savedir)


    return g, frac, df_roc, auc



# MVA monitoring
def PlotTraining(dfs=[], colors=[], titles=[], xtitle='Epoch', ytitle='', filename = 'plot.pdf'):

    # we will get a list of dfs (ideally two) and will create one tgraph each
    graphs = []
    legoptions = []
    plotmin =  999.
    plotmax = -999.
    for i, df in enumerate(dfs):
        graph =  root.TGraph()
        graph.SetLineColor(colors[i])
        graph.SetTitle(titles[i])
        graphs.append(graph)
        legoptions.append('L')

        val = df.as_matrix()
        j=1
        for x in np.nditer(val):
            if x < plotmin:
                plotmin = x
            if x > plotmax:
                plotmax = x
            graph.SetPoint(graph.GetN(), j, x)
            j += 1

    leg = MakeTLegend(xpos=0.4, tobjects=graphs, options=legoptions)

    root.gROOT.SetBatch(1)
    c = MakeTCanvas('c', 'c', 0)
    c.Draw()
    graphs[0].Draw('AL')
    graphs[0].GetYaxis().SetRangeUser(plotmin*0.95, plotmax*1.1)
    graphs[0].GetXaxis().SetTitle(xtitle)
    graphs[0].GetYaxis().SetTitle(ytitle)

    root.gPad.SetGridx()
    root.gPad.SetGridy()

    if len(dfs)==2:
        graphs[1].Draw('L')
    leg.Draw()

    ChangeStyle(c)
    c.Print(filename)

def AddVerticalBox( xposlow, xposhigh, yposlow = -999., yposhigh = -999., color = root.kGray, alpha = 1, style = 1, title =''):

    root.gPad.Update();

    y_low  = root.gPad.GetUymin();
    y_high = root.gPad.GetUymax();

    if yposlow!=-999.:
        y_low = yposlow;
    if yposhigh!=-999.:
        y_high = yposhigh;

    box = root.TBox(xposlow, y_low, xposhigh, y_high);
    box.SetFillColorAlpha(color, alpha);
    box.SetFillStyle(style);
    box.SetLineWidth(1);
    box.SetLineColor(color);

    box.IsA().SetTitle(title);

    return box;

def AddVerticalLine( xpos, yposlow = -999., yposhigh = -999., color = root.kGray, style = 1, width=1, title =''):
    root.gPad.Update();

    y_low  = root.gPad.GetUymin();
    y_high = root.gPad.GetUymax();

    if yposlow!=-999.:
        y_low = yposlow;
    if yposhigh!=-999.:
        y_high = yposhigh;

    line = root.TLine(xpos, y_low, xpos, y_high);
    line.SetLineWidth(width);
    line.SetLineStyle(style);
    line.SetLineColor(color);

    line.IsA().SetTitle(title);

    return line;

def MakePullTCanvas(name, title, entries):
    heightperentry = 27.5
    width  = 250

    w = kCanvasWidth
    h = kCanvasHeight1
    rightmargin  = root.gStyle.GetPadRightMargin();
    leftmargin   = root.gStyle.GetPadLeftMargin();
    bottommargin = root.gStyle.GetPadBottomMargin();
    topmargin    = root.gStyle.GetPadTopMargin();

    totaltopmargin    = topmargin*h;
    totalbottommargin = bottommargin*h;

     # //"size" before applying correction to full width (=wsize)
    totalleftmargin  = leftmargin*w;
    totalrightmargin = rightmargin*w;
    totalwidth       = width + totalrightmargin + totalleftmargin;

      #//need to adjust the left and right margins....
    totalleftmargin  += math.fabs(w-totalwidth)/2.;
    totalrightmargin  += math.fabs(w-totalwidth)/2.;

      #//what is missing to the original size?
    totalwidth += math.fabs(w-totalwidth);

      #//now the total height according to the number of entries
    totalheight = totalbottommargin + totaltopmargin + entries*heightperentry;
    newrightmargin  = totalrightmargin/totalwidth;
    newleftmargin   = totalleftmargin/totalwidth;
    newbottommargin = totalbottommargin/totalheight;
    newtopmargin    = totaltopmargin/totalheight;

    root.gStyle.SetPadRightMargin(newrightmargin);
    root.gStyle.SetPadLeftMargin(newleftmargin);
    root.gStyle.SetPadBottomMargin(newbottommargin);
    root.gStyle.SetPadTopMargin(newtopmargin);

    c = root.TCanvas(name, title, int(totalwidth), int(totalheight));
    c.SetCanvasSize(int(totalwidth), int(totalheight));
    root.gPad.Update();

    #//reset the original gStyle
    root.gStyle.SetPadRightMargin(rightmargin);
    root.gStyle.SetPadLeftMargin(leftmargin);
    root.gStyle.SetPadBottomMargin(bottommargin);
    root.gStyle.SetPadTopMargin(topmargin);

    return c;

def MakePullPlot(name = 'pulls', title = 'title', xtitle ='xtitle', xlow = 0, xhigh =3, names = [], values = (), colors = (), addvalues = True, valuesasint=True):
    # creates a plot with horizontal bars
    n = len(values)

    # set all colors to gray if not given
    while len(colors) < len(values):
        colors.append(root.kGray)

    # set name to integers if not given
    par = len(names) + 1
    while len(names) < len(values):
        names.append('parameter %i' % par)
        par += 1

    c =  MakePullTCanvas(name, title, n);
    c.Draw()

    if max(values) > xhigh:
        xhigh = max(values)*1.05

    histname = name + "hist";
    th1d_frame = MakeTH1D(histname, histname, 1000, xlow, xhigh, xtitle, "", 0); #
    th1d_frame.GetYaxis().SetRangeUser(0., float(n))
    th1d_frame.GetYaxis().SetLabelOffset(999); #//"remove" y-axis label
    th1d_frame.GetYaxis().SetLabelSize(0);# //"remove" y-axis label
    th1d_frame.GetYaxis().SetTickLength(0);# //"remove" y-axis ticks
    th1d_frame.GetXaxis().CenterTitle();
    th1d_frame.Draw("AXIS");
    th1d_frame.GetXaxis().SetNdivisions(505);

    root.gPad.Update()

#      //need NDC coordinates...
    b = c.GetBottomMargin();
    t = c.GetTopMargin();
    r = c.GetRightMargin();
    l = c.GetLeftMargin();
    diff = (1.-t)-b;# //"histogram" height
    step = diff/(float(n)) #//height per entry

    boxes=[]
    texts=[]
    textvalues=[]
    for i in range(n):

        #         //add text, one per value, use same values for the boxes
        x1 = 0.;
        y1 = b+i*step+step/4.;
        x2 = l-0.015;
        y2 = b+i*step+step/4.+step/2.;

        box = AddVerticalBox(min(0, values[i]), max(0, values[i]), 0.25+i, 0.75+i, colors[i], 1.0, 1001, names[i])
        box.SetLineStyle(1);
        box.SetLineColor(root.kBlack);
        box.SetLineWidth(2);
        boxes.append(box)
        box.Draw('LF')

        tp = root.TPaveText(x1, y1, x2, y2,"NDC");
        tp.SetBorderSize(0);
        tp.SetMargin(0);
        tp.SetTextAlign(32);
        tp.SetTextColor(root.kBlack);
        tp.AddText(0.0,0.0, names[i]);
        tp.SetFillColor(0);
        tp.SetFillStyle(0);
        tp.SetTextSize(root.gStyle.GetTitleSize("X")*0.8);
        texts.append(tp)
        tp.Draw();

        x1 = 1.-r+0.015;
        x2 = 1.;

        if addvalues:
            tpv = root.TPaveText(x1, y1, x2, y2,"NDC")
            tpv.SetBorderSize(0)
            tpv.SetMargin(0)
            tpv.SetTextAlign(12);
            tpv.SetTextColor(root.kBlack)
            if valuesasint:
                tpv.AddText(0.0, 0.0, ' %i' % int(values[i]))
            else:
                tpv.AddText(0.0, 0.0, ' %3.3f' % values[i])
            textvalues.append(tpv)
            tpv.SetFillColor(0);
            tpv.SetFillStyle(0);
            tpv.SetTextSize(root.gStyle.GetTitleSize("X")*0.8);
            tpv.AppendPad();

    return c, th1d_frame, boxes, texts, textvalues



def getNSKFitRange(hist, frac=0.5, frachigh=0.5, smooth=0):
    if(frachigh<0):
        frachigh = frac;

    histclone = hist.Clone();
    histclone.Smooth(smooth+1);

    mean = 999.;
    rangemin = 999.;
    rangemax = -999.;

    binmax        = histclone.GetMaximumBin();
    xmaxbincenter = histclone.GetXaxis().GetBinCenter(binmax);
    xmaxvalue     = histclone.GetBinContent(binmax);
    rms           = histclone.GetRMS();

    # find lower end
    i = binmax
    while i > 1:
        if histclone.GetBinContent(i) < frac*xmaxvalue:
            rangemin = histclone.GetBinCenter(i)
            break
        i -= 1

    i = binmax
    while i <= histclone.GetNbinsX():
        if histclone.GetBinContent(i) < frachigh*xmaxvalue:
            rangemax = histclone.GetBinCenter(i)
            break
        i += 1

    return xmaxbincenter, rangemin, rangemax


def fcn_crystalball(x, par):

    norm = par[0]
    alpha = par[1]
    n = par[2]
    sigma = par[3]
    mean = par[4]

    if (sigma < 0.):
        return 0.

    z = (x[0] - mean)/sigma

    if (alpha < 0):
        z = -z

    abs_alpha = math.fabs(alpha)

    if (z  > - abs_alpha):
        return norm*math.exp(- 0.5 * z * z)
    else:
        nDivAlpha = n / abs_alpha
        AA =  math.exp(-0.5*abs_alpha*abs_alpha)
        B = nDivAlpha - abs_alpha
        arg = nDivAlpha/(B-z)
        return norm * AA * math.pow(arg,n)


def fcn_novosibirsk(x, par):
    qa=0
    qb=0
    qc=0
    qx=0
    qy=0

    peak = par[1];
    width = par[2];
    sln4 = math.sqrt(math.log(4));
    y = par[3]*sln4;
    tail = -math.log(y + math.sqrt(1 + y*y))/sln4;

    if(math.fabs(tail) < 1e-7):
        qc = 0.5*((x[0]-peak)*(x[0]-peak)/width/width)
    else:
        qa = tail*sln4;
        qb = math.sinh(qa)/qa;
        qx = 1*(x[0]-peak)/width*qb; #// "The -1 here swaps the X-axis from Chris' original." -> swapped back (Torben)
        qy = 1.0+tail*qx;

        if( qy > 1e-7 ):
            lqt = math.log(qy)/tail
            qc = 0.5*(lqt*lqt + tail*tail)
        else:
            qc = 15.0

    y = par[0]*math.exp(-qc);

    return y


# from Danika and Belina
# Function returns the index of the value in an array nearest to the input value
def find_nearest(vector, value):
    idx = (np.abs(vector-value)).argmin()
    return idx

# Function to iteratively find the appropriate signal floor for a given background
# level, at a given confidence level
def signalFloor(Bg, conf=0.9, accuracy=0.01, bgonly = False):

    # Convert the confidence level to a cdf:
    conf = 1-conf

    # First, calculate the expectation with a background-only hypothesis
    import scipy.stats as st
    floor_bg = Bg - st.poisson.ppf(conf, Bg)

    if bgonly == True:
        return floor_bg

    # Make an interval around floor_bg
    interval = np.linspace(0.5*floor_bg, max(2*floor_bg, 5.0), 1000)

    # Reduce the interval iteratively until reaching desired
    deviations = ( conf - st.poisson.cdf(round(Bg), Bg+interval) )

    # Now, iteratively reduce the interval until the desired decimal precision is reached
    while np.max(interval)-np.min(interval) > accuracy:
        upper_lim = interval[np.all( [ (deviations > 0), (deviations <= np.min(deviations[deviations > 0])) ], axis=0 )]

        lower_lim = interval[np.all( [ (deviations < 0), (deviations >= np.max(deviations[deviations < 0])) ], axis=0 )]

        interval = np.linspace(lower_lim, upper_lim, 1000)
        deviations = ( conf - st.poisson.cdf(round(Bg), Bg+interval) )

    return np.mean(interval)


def AddHorizontalLine( ypos, color=root.kBlack, style=1, width=1, title='line'):

    root.gPad.Update()
    x_low_  = root.gPad.GetUxmin()
    x_high_ = root.gPad.GetUxmax()

    line = root.TLine(x_low_, ypos, x_high_,ypos)
    line.SetLineColor(color)
    line.SetLineStyle(style)
    line.SetLineWidth(width)

    line.IsA().SetTitle(title);

    return line


def DrawResiduals(cname, hist, tg, low, high, savedir='', restype='normres', text='', texts=[], xtexts=[], ytexts1=[], ytexts2=[]):

    # get range of tg
    tg_n = tg.GetN()
    tg_x = tg.GetX()

    fitlow = 999.0
    fithigh = -999
    for i in range( tg_n ):
        if tg_x[i] < fitlow:
            fitlow = tg_x[i]

        if tg_x[i] > fithigh:
            fithigh = tg_x[i]

    # make canvas
    c = MakeTCanvas(cname, cname, 1);
    c.cd()

    # clone histogram
    histclone  = hist.Clone()
    histclone.SetName("th1_bottom1")
    histclone.SetDirectory(0)

    # make pads
    h1 = kCanvasHeight1;
    h2 = kCanvasHeight2;
    bottom_for_title  = root.gStyle.GetPadBottomMargin()*h1;
    bottom_pad_height = (h2-h1+root.gStyle.GetPadBottomMargin()*h1-1);
    top_down_coordinate = (h2-h1)/h2;
    bottom_up_coordinate = bottom_pad_height/h2; #the "-1" takes care for the linewidth of 2 (which is draw below(!) the axis by ROOT
    kRatioTitleOffsetTopY = 1.6*1.1;

    #TOP PAD
    c.cd();
    tp_top = root.TPad("tp_top", "tp_top", 0, top_down_coordinate, 1, 1);
    tp_top.Draw();
    tp_top.cd();

    hist.Draw("E");
    tg.Draw("L");

    #dirty part: cover the "zero" of the top histogram
    cover = root.TPaveText(0.01, 0.1, 0.175, 0.25, "NDC")
    cover.SetLineColor(root.kWhite)
    cover.SetFillColor(root.kWhite)
    cover.SetBorderSize(0)
    cover.SetShadowColor(root.kWhite)
    cover.Draw()
    tp_top.Update()
    root.gPad.RedrawAxis()

    cover2 = root.TPaveText(0.91, 0.1, 1.0, 0.25, "NDC")
    cover2.SetLineColor(root.kWhite)
    cover2.SetFillColor(root.kWhite)
    cover2.SetBorderSize(0)
    cover2.SetShadowColor(root.kWhite)
    cover2.Draw()
    root.gPad.SetGridx()
    root.gPad.SetGridy()
    tp_top.Update()
    root.gPad.RedrawAxis()

    leg = MakeTLegend(xpos=0.25, ypos=0.9, title='', tobjects=[hist, tg], options=['LEP','L'])
    leg.Draw()

    tinfo = MakeTPText(text=text, x1 = 0.3)
    tinfo.Draw()

    uotext = GetUnderOverFlow(hists=[hist])[0]
    uotext.Draw()

    # extra text
    t = []
    for i in range(0, len(texts), 1):
        print(i, texts[i])

        t_text = MakeTPText(text=texts[i], x1 = xtexts[i], y1=ytexts1[i], y2=ytexts2[i])
        t_text.Draw()
        t.append(t_text)

    # BOTTOM PAD
    c.cd()
    tp_bottom = root.TPad("tp_bottom","tp_bottom",0,0,1,bottom_up_coordinate)
    tp_bottom.SetBottomMargin(bottom_for_title/bottom_pad_height)
    tp_bottom.SetTopMargin(5/bottom_pad_height)
    tp_bottom.Draw()
    tp_bottom.cd()

    histclone.GetXaxis().SetTitleOffset(3.8)
    histclone.GetYaxis().SetTitleOffset(1.6*1.1)
    histclone.GetYaxis().SetNdivisions(304)

    tg_residuals = root.TGraph()
    tg_residuals.SetMarkerColor(hist.GetMarkerColor())

    if(restype=="normres"):

        hist_pull = MakeTH1D('hist_pull', 'hist_pull', int((high-low)*4), 2*low, 2*high, 'R / #sigma', 'Entries / bin')
        #hist = MakeTH1D( 'hist_%i'%i, title, nbins, low, high, xtitle, ytitle, linecolor=colors[i])

        for i in range(histclone.GetNbinsX()):
            xval    = histclone.GetBinCenter(i+1);
            valhist = histclone.GetBinContent(i+1);
            valerrhist = histclone.GetBinError(i+1);
            valfit  = tg.Eval(xval);

            valpoint = 0.
            if(valerrhist>1e-9 and xval>fitlow and xval<fithigh):
                valpoint = (valhist-valfit)/valerrhist;
                tg_residuals.SetPoint(tg_residuals.GetN(), xval, valpoint);
                hist_pull.Fill(valpoint)

        histclone.GetYaxis().SetTitle("R / #sigma");
        histclone.GetYaxis().SetRangeUser(low,high);

    tp_bottom.cd()
    histclone.Reset()
    histclone.SetLineWidth(0)

    histclone.Draw('')
    tg_residuals.Draw('P')
    root.gPad.SetGridx()
    root.gPad.SetGridy()

    tp_top.cd();

    SavePdf(c, savedir=savedir)

    cpull = MakeTCanvas('%s_pull'%cname, '%s_pull'%cname, 0);
    cpull.cd()
    hist_pull.Draw('HIST')
    SavePdf(cpull, savedir=savedir)

    return 0


def getFracIntegral(h, xmin, xmax):
    axis = h.GetXaxis()
    bmin = axis.FindBin(xmin)
    bmax = axis.FindBin(xmax)
    integral = h.Integral(bmin,bmax)
    if bmin>0:
        integral -= h.GetBinContent(bmin)*(xmin-axis.GetBinLowEdge(bmin))/axis.GetBinWidth(bmin)
    if bmax<h.GetNbinsX()+1:
        integral -= h.GetBinContent(bmax)*(axis.GetBinUpEdge(bmax)-xmax)/axis.GetBinWidth(bmax)

    return integral


