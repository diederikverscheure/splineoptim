import sys
from numpy import *
#from pulp import *
from cvxopt import solvers,matrix,sparse,umfpack
from scipy.misc import factorial
import wx
import time
import wx.lib.plot as plot
import wx.lib.scrolledpanel as scrolled
from matplotlib.pyplot import figure, show

class SplineOptim:
    def __init__(self):
        self.constraints = array([], dtype=float)
        self.objectives = array([], dtype=float)
        # units, RPM, start time, end time, grid size, cont level,
        # bound cond type, bound cond level, solver, output
        self.SetOptions(0, 0, 0, 10, 50, 2, 0, 2, 0, 0)
        self.AddConstraint(0, self.ts, self.ts, 1, 1)
        self.AddConstraint(0, self.te/2.0, self.te/2.0, 2, 2)
        self.AddConstraint(0, self.ts, self.te, 1, 2)
        self.AddObjective(2, self.M+1, 1)
        self.AddObjective(0, self.M+2, 0.01)
        self.xs = zeros((self.K+1,self.M+2))

    def GetNames(self):
        names = ["Position", "Velocity", "Acceleration", "Jerk", "Jolt"]
        for i in range(5,int(self.M)+2):
            name = str(i) + "th derivative"
            names = names + [name]
        return names[0:int(self.M)+3]
    
    def GetNorms(self):
        return ["One-norm", "Two-norm", "Inf-norm", "Max. abs", "Min. abs"]

    def GetUnits(self):
        return self.units        
        
    def GetRPM(self):
        return self.RPM

    def GetStartTime(self):
        return self.ts
        
    def GetEndTime(self):
        return self.te

    def GetGridSize(self):
        return self.K
        
    def GetContLevel(self):
        return self.M
                
    def GetBoundCondType(self):
        return self.bctype

    def GetBoundCondLevel(self):
        return self.N

    def GetSolver(self):
        return self.solver
    
    def GetOutputLevel(self):
        return self.output
                
    def GetTime(self):
        return linspace(self.ts,self.te,self.K+1)
    
    def GetSolution(self, k):
        if k <= self.xs.shape[1]:
            return self.xs[:,k]
            
    def GetConstraints(self):
        return self.constraints

    def GetObjectives(self):
        return self.objectives

    def SetOptions(self, units, RPM, ts, te, K, M, bctype, N, solver, output):
        self.units = units
        self.RPM = RPM
        self.ts = ts
        self.te = te
        self.K = int(K)
        self.M = int(M)
        self.bctype = bctype
        self.N = int(min(N,M))
        self.solver = solver
        self.output = output
        for i in range(self.GetNrConstraints()):
            self.SetConstrValue(i, min(self.GetConstrValue(i), self.GetContLevel()+3))
        for i in range(self.GetNrObjectives()):
            self.SetObjValue(i, min(self.GetObjValue(i), self.GetContLevel()+3))
        if output == 0:
            solvers.options['show_progress'] = False
        else:
            solvers.options['show_progress'] = True

    def GetNrConstraints(self):
        return self.constraints.shape[0]
        
    def GetConstrValue(self, i):
        if i <= self.GetNrConstraints():
            return int(self.constraints[i,0])
    
    def SetConstrValue(self, i, k):
        if i <= self.GetNrConstraints():
            self.constraints[i,0] = k
        
    def GetFromTime(self, i):
        if i <= self.GetNrConstraints():
            return self.constraints[i,1]
        
    def SetFromTime(self, i, tl):
        if i <= self.GetNrConstraints():
            self.constraints[i,1] = tl

    def GetToTime(self, i):
        if i <= self.GetNrConstraints():
            return self.constraints[i,2]

    def SetToTime(self, i, tu):
        if i <= self.GetNrConstraints():
            self.constraints[i,2] = tu
            
    def GetFromValue(self, i):
        if i <= self.GetNrConstraints():
            return self.constraints[i,3]

    def SetFromValue(self, i, xl):
        if i <= self.GetNrConstraints():
            self.constraints[i,3] = xl
            
    def GetToValue(self, i):
        if i <= self.GetNrConstraints():
            return self.constraints[i,4]

    def SetToValue(self, i, xu):
        if i <= self.GetNrConstraints():
            self.constraints[i,4] = xu

    def GetTotalConstrCost(self):
            return sum(self.constraints[:,5])

    def GetConstrCost(self, i):
        if i <= self.GetNrConstraints():
            return self.constraints[i,5]
            
    def SetConstrCost(self, i, c):
        if i <= self.GetNrConstraints():
            self.constraints[i,5] = c

    def AddConstraint(self, k, tl, tu, xl, xu):
        self.UpdateConstraint(self.GetNrConstraints()+1, k, tl, tu, xl, xu)

    def RemoveConstraint(self, i):
        if i < self.GetNrConstraints() and i >= 0:
            j = setdiff1d(array(range(self.GetNrConstraints())),array([i]))
            self.constraints = self.constraints[j,:]
        
    def UpdateConstraint(self, i, k, tl, tu, xl, xu):
        if tl > tu:
            tt = tl
            tl = tu
            tu = tt
        if xl > xu:
            xt = xl
            xl = xu
            xu = xt
        if i <= self.GetNrConstraints():
            self.constraints[i,0] = k
            self.constraints[i,1] = tl
            self.constraints[i,2] = tu
            self.constraints[i,3] = xl
            self.constraints[i,4] = xu
            self.constraints[i,5] = 0
        else:
            if self.GetNrConstraints() == 0:
                self.constraints = array([[k, tl, tu, xl, xu, 0]], dtype=float)
            else:
                constraint = array([k, tl, tu, xl, xu, 0], dtype=float)
                self.constraints = vstack((self.constraints, constraint))

    def GetNrObjectives(self):
        return self.objectives.shape[0]    
    
    def GetObjNorm(self,i):
        if i <= self.GetNrObjectives():
            return self.objectives[i,0]

    def SetObjNorm(self,i,n):
        if i <= self.GetNrObjectives():
            self.objectives[i,0] = n
            
    def GetObjValue(self,i):
        if i <= self.GetNrObjectives():
            return self.objectives[i,1]

    def SetObjValue(self,i,k):
        if i <= self.GetNrObjectives():
            self.objectives[i,1] = k
            
    def GetObjWeight(self,i):
        if i <= self.GetNrObjectives():
            return self.objectives[i,2]

    def SetObjWeight(self,i,w):
        if i <= self.GetNrObjectives():
            self.objectives[i,2] = w
            
    def AddObjective(self, n, k, w):
        self.UpdateObjective(self.GetNrObjectives()+1, n, k, w)
        
    def RemoveObjective(self, i):
        if i < self.GetNrObjectives() and i >= 0:
            j = setdiff1d(array(range(self.GetNrObjectives())),array([i]))
            self.objectives = self.objectives[j,:]

    def UpdateObjective(self, i, n, k, w):
        if i <= self.GetNrObjectives():
            self.SetObjNorm(i,n)
            self.SetObjValue(i,k)
            self.SetObjWeight(i,w)
        else:
            if self.GetNrObjectives() == 0:
                self.objectives = array([[n, k, w]], dtype=float)
            else:
                objective = array([n, k, w], dtype=float)
                self.objectives = vstack((self.objectives, objective))

    def SetupProb(self):
        Ai = self.GetIntegration()
        bi = zeros((Ai.shape[0],1))
        Ab = self.GetBoundCond()
        bb = zeros((Ab.shape[0],1))
        Aeq = vstack((Ai,Ab))
        beq = vstack((bi,bb))
        Aineq = zeros((0,(self.M+1)*(self.K+1)))
        bineq = zeros((Aineq.shape[0],1))
        bs = zeros((0,1))
        c = zeros(((self.K+1)*(self.M+1),1))        
        t = self.GetTime()
        for i in range(self.GetNrObjectives()):
            n = self.GetObjNorm(i)
            k = self.GetObjValue(i)
            w = self.GetObjWeight(i)
            As = self.GetSlackCond(k)
            if n == 0:
                # one-norm
                c = vstack((c,w*ones((As.shape[0],1))/As.shape[0]))
                Az = zeros((As.shape[0],Aineq.shape[1]-(self.M+1)*(self.K+1)))
                Ao1 = hstack((-As,Az,-eye(As.shape[0])))
                Ao2 = hstack((As,Az,-eye(As.shape[0])))
                Ao = vstack((Ao1,Ao2))
            elif n == 1:
                # two-norm
                Ao = zeros((0,Aineq.shape[1]))
            elif n == 2:
                # inf-norm
                c = vstack((c,w))
                Az = zeros((As.shape[0],Aineq.shape[1]-(self.M+1)*(self.K+1)))
                Ao1 = hstack((-As,Az,-ones((As.shape[0],1))))
                Ao2 = hstack((As,Az,-ones((As.shape[0],1))))
                Ao = vstack((Ao1,Ao2))
            Aineq = vstack((hstack((Aineq,zeros((Aineq.shape[0],Ao.shape[1]-Aineq.shape[1])))),Ao))
            bineq = vstack((bineq,zeros((Ao.shape[0],1))))
        bs = vstack((bs,bineq.shape[0]))
        for i in range(self.GetNrConstraints()):
            k = self.GetConstrValue(i)
            tl = self.GetFromTime(i)
            tu = self.GetToTime(i)
            xl = self.GetFromValue(i)
            xu = self.GetToValue(i)
            ti = hstack((tl,t[(t>tl)*(t<tu)],tu))
            ti = unique(setdiff1d(ti,[-inf, inf]))
            Ac = zeros((ti.shape[0],(self.M+1)*(self.K+1)))
            idx = max(0,k-1)*(self.K+1) + interp(ti,t,range(t.shape[0]))
            for j in range(ti.shape[0]):
                if round(idx[j]) == idx[j]:
                    Ac[j,int(idx[j])] = 1
                else:
                    Ac[j,int(floor(idx[j]))] = (ceil(idx[j])-idx[j])
                    Ac[j,int(ceil(idx[j]))] = (idx[j]-floor(idx[j]))
            if isfinite(xl) and isfinite(xu) and xl == xu:
                Aeq = vstack((Aeq,Ac))
                beq = vstack((beq,xl*ones((ti.shape[0],1))))
            else:
                if isfinite(xl):
                    Aineq = vstack((Aineq,hstack((-Ac,zeros((Ac.shape[0],Aineq.shape[1]-Ac.shape[1]))))))
                    bineq = vstack((bineq,-xl*ones((ti.shape[0],1))))
                if isfinite(xu):
                    Aineq = vstack((Aineq,hstack((Ac,zeros((Ac.shape[0],Aineq.shape[1]-Ac.shape[1]))))))
                    bineq = vstack((bineq,xu*ones((ti.shape[0],1))))
            bs = vstack((bs,(xl != xu)*(isfinite(xl)*1+isfinite(xu)*1)*Ac.shape[0]))
        Aeq = hstack((Aeq,zeros((Aeq.shape[0],Aineq.shape[1]-Aeq.shape[1]))))

        # Presolve integration equations
        idx = hstack((range(0,self.M*(self.K+1),self.K+1),range(self.M*(self.K+1),(self.M+1)*(self.K+1))))
        didx = setdiff1d(range(0,(self.M+1)*(self.K+1)),idx)
        Ar = matrix(-Ai[:,idx])
        Al = sparse(matrix(Ai[:,didx]))
        umfpack.linsolve(Al,Ar)
        
        Ar = array(Ar)
        At = zeros((Ai.shape[1],Ar.shape[1]))
        At[didx,:] = Ar
        At[idx,:] = eye(Ar.shape[1])
        At = hstack((At,zeros((At.shape[0],Aeq.shape[1]-Ai.shape[1]))))
        At = vstack((At,hstack((zeros((Aeq.shape[1]-Ai.shape[1],Ar.shape[1])),eye(Aeq.shape[1]-Ai.shape[1])))))
        self.At = At
        
        Aeq = dot(Aeq,At)
        Aeq = Aeq[Ai.shape[0]:,:]
        beq = beq[Ai.shape[0]:,:]
        Aineq = dot(Aineq,At)
        c = dot(transpose(At),c)
        
        self.Aeq = sparse(matrix(Aeq))
        self.beq = matrix(beq)
        self.Aineq = sparse(matrix(Aineq))
        self.bineq = matrix(bineq)
        self.c = matrix(c)
        self.bs = bs
    
    def SolveProb(self):
        self.SetupProb()
        t1 = time.time()
        sol = solvers.lp(self.c,self.Aineq,self.bineq,self.Aeq,self.beq)
        t2 = time.time()
        xs = array(sol.get('x'))
        ys = array(sol.get('y'))
        ss = array(sol.get('s'))
        for i in range(self.GetNrConstraints()):
            if self.bs[i+1] != 0:
                self.SetConstrCost(i, sum(ss[range(int(sum(self.bs[0:i+1])),int(sum(self.bs[0:i+1])+self.bs[i+1]))]))
        status = sol.get('status') == 'optimal'
        xs = dot(self.At,xs)
        self.xs = reshape(xs[0:(self.M+1)*(self.K+1)],(self.K+1,self.M+1),'F')
        t = self.GetTime()
        xM1 = hstack((0,diff(self.xs[:,self.M],1,0)/diff(t),0))
        tm = hstack((t[0],t[0:-1]/2+t[1:]/2,t[-1]))
        xM1 = reshape(interp(t,tm,xM1),(-1,1))
        xM2 = reshape(hstack((0,diff(self.xs[:,self.M],2,0)/diff(t[0:-1])/diff(t[1:]),0)),(-1,1))
        self.xs = hstack((self.xs,xM1,xM2))
        return (status,t2-t1)
    
    def GetIntegration(self):
        dt = diff(self.GetTime())
        Ai = zeros((0,(self.M+1)*(self.K+1)))
        for n in range(1, self.M+1):
            An1 = hstack((diag(n*dt**n/factorial(n+1),0),zeros((self.K,1))))
            An1 = An1 + hstack((zeros((self.K,1)),diag(dt**n/factorial(n+1),0)))
            An2 = empty((self.K,0))
            for nv in range(1,n):
                An2 = hstack((An2,diag(dt**nv/factorial(nv),0),zeros((self.K,1))))
            An3 = hstack((eye(self.K),zeros((self.K,1))))
            An3 = An3 + hstack((zeros((self.K,1)),-eye(self.K)))            
            An4 = zeros((self.K,(self.M-n)*(self.K+1)))
            An = hstack((An4,An3,An2,An1))
            Ai = vstack((Ai,An))
        return Ai
    
    def GetBoundCond(self):
        if self.bctype == 0:
            # Zero
            Ab = zeros((2*self.N,(self.M+1)*(self.K+1)))
            for i in range(1,self.N+1):
                Ab[2*i-2,i*(self.K+1)] = 1
                Ab[2*i-1,(i+1)*(self.K+1)-1] = 1
        elif self.bctype == 1:
            # Periodic
            Ab = zeros((self.N,(self.M+1)*(self.K+1)))
            for i in range(1,self.N+1):
                Ab[i-1,i*(self.K+1)] = 1
                Ab[i-1,(i+1)*(self.K+1)-1] = -1
        elif self.bctype == 2:
            # Symmetric
            Ab = zeros((self.N+1,(self.M+1)*(self.K+1)))
            for i in range(0,self.N+1):
                Ab[i,i*(self.K+1)] = 1
                Ab[i,(i+1)*(self.K+1)-1] = (-1.0)**(i+1)         
        return Ab

    def GetSlackCond(self,k):
        if k <= self.M:
            As = hstack((zeros((self.K+1,int(max(0,k))*(self.K+1))),eye(self.K+1),zeros((self.K+1,(self.M-int(max(0,k)))*(self.K+1)))))
        elif k == self.M+1:
            dt = diff(self.GetTime())
            Ap = hstack((diag(-1/dt,0),zeros((self.K,1))))
            Ap = Ap + hstack((zeros((self.K,1)),diag(1/dt,0)))
            As = hstack((zeros((self.K,self.M*(self.K+1))),Ap))
        elif k == self.M+2:
            dt = diff(self.GetTime())    
            App = hstack((diag(1/dt[0:-1]/dt[1:],0),zeros((self.K-1,2))))
            App = App + hstack((zeros((self.K-1,1)),-2*diag(1/dt[0:-1]/dt[1:],0),zeros((self.K-1,1))))
            App = App + hstack((zeros((self.K-1,2)),diag(1/dt[0:-1]/dt[1:],0)))
            As = hstack((zeros((self.K-1,self.M*(self.K+1))),App))
        return As        

class MainFrame(wx.Frame):
    def __init__(self):
        self.SplineOptim = SplineOptim()
        #self.SplineOptim.SetupProb()
        wx.Frame.__init__(self, None, -1, "SplineOptim",size=wx.Size(1000,760))
        # Window size
        self.CentreOnScreen()
        self.SetFocus()
        self.Maximize(True)
        self.SetupPanels()
        self.SetupMenus()
        self.SetupSolveOptions()
        self.SetupConstraints()
        self.SetupObjectives()
        self.PlotSolution()

    def SetupPanels(self):
        self.winSize = self.GetSize()
        self.layoutFlags = wx.ALL | wx.EXPAND | wx.ALIGN_CENTER
        self.border = 2
        self.optionsSize = 350.0
        self.minHeight = 28
        self.plotSize = (self.winSize[0]-self.optionsSize-10,int(0.5*(self.winSize[0]-self.optionsSize-10)))

        # Tab panels
        bkMain = wx.Notebook(self,style=wx.BK_DEFAULT)
        self.pnlMotionLaw = wx.Panel(bkMain)
        self.pnlPathTracking = wx.Panel(bkMain)
        bkMain.AddPage(self.pnlMotionLaw,"Motion law")
        #bkMain.AddPage(self.pnlPathTracking,"Path tracking")
        
        # Left and right panel
        self.pnlOptions = scrolled.ScrolledPanel(self.pnlMotionLaw, -1)
        self.pnlOptions.SetupScrolling(scroll_x=False)
        self.pnlPlot = scrolled.ScrolledPanel(self.pnlMotionLaw, -1)
        self.pnlPlot.SetupScrolling(scroll_x=False)

        bsMain = wx.BoxSizer(wx.HORIZONTAL)
        bsMain.Add(self.pnlOptions, 0, self.layoutFlags, self.border)
        bsMain.Add(self.pnlPlot, 1, self.layoutFlags, self.border)
        self.pnlMotionLaw.SetSizer(bsMain)

        self.pnlSolveOptions = wx.Panel(self.pnlOptions, -1, name="Solver options")
        self.bsSolveOptions = wx.StaticBoxSizer(wx.StaticBox(self.pnlSolveOptions, -1, "Solver options"), wx.VERTICAL)
        self.pnlSolveOptions.SetSizer(self.bsSolveOptions)
        self.pnlConstraints = wx.Panel(self.pnlOptions, -1, name="Constraints")
        self.bsConstraints = wx.StaticBoxSizer(wx.StaticBox(self.pnlConstraints, -1, "Constraints"), wx.VERTICAL)
        self.pnlConstraints.SetSizer(self.bsConstraints)
        self.pnlObjectives = wx.Panel(self.pnlOptions, -1, name="Objective functions")
        self.bsObjectives = wx.StaticBoxSizer(wx.StaticBox(self.pnlObjectives, -1, "Objective functions"), wx.VERTICAL)
        self.pnlObjectives.SetSizer(self.bsObjectives)        
        self.bsOptions = wx.BoxSizer(wx.VERTICAL)
        self.bsOptions.Add(self.pnlSolveOptions, 0, self.layoutFlags, self.border)
        self.bsOptions.Add(self.pnlConstraints, 0, self.layoutFlags, self.border)
        self.bsOptions.Add(self.pnlObjectives, 0, self.layoutFlags, self.border)
        self.pnlOptions.SetSizer(self.bsOptions)
        
        self.bsPlot = wx.BoxSizer(wx.VERTICAL)
        self.pcPosition = plot.PlotCanvas(self.pnlPlot)
        self.pcPosition.SetInitialSize(self.plotSize)
        self.pcPosition.SetEnableGrid(True)

        self.pcVelocity = plot.PlotCanvas(self.pnlPlot)
        self.pcVelocity.SetInitialSize(self.plotSize)
        self.pcVelocity.SetEnableGrid(True)

        self.pcAcceleration = plot.PlotCanvas(self.pnlPlot)
        self.pcAcceleration.SetInitialSize(self.plotSize)
        self.pcAcceleration.SetEnableGrid(True)

        self.bsPlot.Add(self.pcPosition, 0, self.layoutFlags, self.border)
        self.bsPlot.Add(self.pcVelocity, 0, self.layoutFlags, self.border)
        self.bsPlot.Add(self.pcAcceleration, 0, self.layoutFlags, self.border)
        self.pnlPlot.SetSizer(self.bsPlot)
        
    def SetupMenus(self):
        # Menu and statusbar 
        self.statusBar = self.CreateStatusBar()
        menuBar = wx.MenuBar()
        self.SetMenuBar(menuBar)
        mnuFile = wx.Menu()
        
        menuBar.Append(mnuFile, "&File")
        mnuFile.Append(101, "L&oad file\tCtrl-O", "Load file")
        mnuFile.Append(102, "&Save file\tCtrl-S", "Save file")        
        mnuFile.Append(wx.ID_EXIT, "&Quit\tCtrl-Q", "Exit")
        self.Bind(wx.EVT_MENU, self.OnLoad, id=101)
        self.Bind(wx.EVT_MENU, self.OnSave, id=102)
        self.Bind(wx.EVT_MENU, self.OnQuit, id=wx.ID_EXIT)        

        mnuSolve = wx.Menu()
        menuBar.Append(mnuSolve, "&Solve")
        mnuSolve.Append(201, "&Solve")
        self.Bind(wx.EVT_MENU, self.OnSolve, id=201)

        mnuHelp = wx.Menu()
        menuBar.Append(mnuHelp, "&Help")
        mnuHelp.Append(301, "&About")
        self.Bind(wx.EVT_MENU, self.OnAbout, id=301)

    def SetupSolveOptions(self):
        # Solveroptions static box
        bsSolveOptions1 = wx.FlexGridSizer(0,3,2,2)
        bsSolveOptions2 = wx.FlexGridSizer(0,4,2,2)
        
        for c in (bsSolveOptions1,bsSolveOptions2):
            self.bsSolveOptions.Add(c, 0, self.layoutFlags)

        self.minWidth = int(self.optionsSize/4.0)
        self.btnSolve = wx.Button(self.pnlSolveOptions, -1, "Solve")
        self.btnLoad = wx.Button(self.pnlSolveOptions, -1, "Load")
        self.btnSave = wx.Button(self.pnlSolveOptions, -1, "Save")
        font = self.btnSolve.GetFont()
        font.SetWeight(wx.FONTWEIGHT_BOLD)
        self.btnSolve.SetFont(font)
        self.btnSolve.Bind(wx.EVT_BUTTON, self.OnSolve, id=self.btnSolve.GetId())
        self.btnLoad.Bind(wx.EVT_BUTTON, self.OnLoad, id=self.btnLoad.GetId())
        self.btnSave.Bind(wx.EVT_BUTTON, self.OnSave, id=self.btnSave.GetId())

        self.stUnits = wx.StaticText(self.pnlSolveOptions, -1, "Units")
        self.chUnits = wx.Choice(self.pnlSolveOptions, -1, choices=["time", "degrees"])
        self.chUnits.SetSelection(self.SplineOptim.GetUnits())

        self.stRPM = wx.StaticText(self.pnlSolveOptions, -1, "RPM")
        self.edtRPM = wx.TextCtrl(self.pnlSolveOptions, -1, str(self.SplineOptim.GetRPM()))
                
        self.stStartTime = wx.StaticText(self.pnlSolveOptions, -1, "Start time")
        self.edtStartTime = wx.TextCtrl(self.pnlSolveOptions, -1, str(self.SplineOptim.GetStartTime()))

        self.stEndTime = wx.StaticText(self.pnlSolveOptions, -1, "End time")
        self.edtEndTime = wx.TextCtrl(self.pnlSolveOptions, -1, str(self.SplineOptim.GetEndTime()))

        self.stGridSize = wx.StaticText(self.pnlSolveOptions, -1, "Grid size")
        self.edtGridSize = wx.TextCtrl(self.pnlSolveOptions, -1, str(self.SplineOptim.GetGridSize()))

        self.stContLevel = wx.StaticText(self.pnlSolveOptions, -1, "Cont. level")
        self.edtContLevel = wx.TextCtrl(self.pnlSolveOptions, -1, str(self.SplineOptim.GetContLevel()))

        self.stBoundCondType = wx.StaticText(self.pnlSolveOptions, -1, "Bound. cond")
        self.chBoundCondType = wx.Choice(self.pnlSolveOptions, -1, choices=["Zero", "Periodic", "Symmetric"])
        self.chBoundCondType.SetSelection(self.SplineOptim.GetBoundCondType())

        self.stBoundCondLevel = wx.StaticText(self.pnlSolveOptions, -1, "Bound. level")
        self.edtBoundCondLevel = wx.TextCtrl(self.pnlSolveOptions, -1, str(self.SplineOptim.GetBoundCondLevel()))

        self.stSolver = wx.StaticText(self.pnlSolveOptions, -1, "Solver")
        self.chSolver = wx.Choice(self.pnlSolveOptions, -1, choices=["Cvxopt", "Clp"])
        self.chSolver.SetSelection(self.SplineOptim.GetSolver())

        self.stOutputLevel = wx.StaticText(self.pnlSolveOptions, -1, "Output level")        
        self.chOutputLevel = wx.Choice(self.pnlSolveOptions, -1, choices=["None", "Verbose"])
        self.chOutputLevel.SetSelection(self.SplineOptim.GetOutputLevel())
        
        self.chUnits.Bind(wx.EVT_CHOICE, self.OnSolveOptions)
        self.edtRPM.Bind(wx.EVT_KILL_FOCUS, self.OnSolveOptions)
        self.edtStartTime.Bind(wx.EVT_KILL_FOCUS, self.OnSolveOptions)
        self.edtEndTime.Bind(wx.EVT_KILL_FOCUS, self.OnSolveOptions)
        self.edtGridSize.Bind(wx.EVT_KILL_FOCUS, self.OnSolveOptions)
        self.edtContLevel.Bind(wx.EVT_KILL_FOCUS, self.OnSolveOptions)
        self.chBoundCondType.Bind(wx.EVT_CHOICE, self.OnSolveOptions)
        self.edtBoundCondLevel.Bind(wx.EVT_KILL_FOCUS, self.OnSolveOptions)
        self.chSolver.Bind(wx.EVT_CHOICE, self.OnSolveOptions)
        self.chOutputLevel.Bind(wx.EVT_CHOICE, self.OnSolveOptions)
        
        for c in (self.btnSolve,self.btnLoad,self.btnSave):
            c.SetMinSize((self.optionsSize/3.0-self.border*2.0, self.minHeight))
            bsSolveOptions1.Add(c, 0, self.layoutFlags, self.border)
        
        for c in (self.stUnits,self.chUnits,self.stRPM,self.edtRPM,
                  self.stStartTime,self.edtStartTime,self.stEndTime,self.edtEndTime,
                  self.stGridSize,self.edtGridSize,self.stContLevel,self.edtContLevel,
                  self.stBoundCondType,self.chBoundCondType,self.stBoundCondLevel,self.edtBoundCondLevel,
                  self.stSolver,self.chSolver,self.stOutputLevel,self.chOutputLevel):
            c.SetMinSize((self.optionsSize/4.0-self.border*2.0, self.minHeight))
            bsSolveOptions2.Add(c, 0, self.layoutFlags, self.border)
        
        self.ShowSolveOptions()
        
    def SetupConstraints(self):
        bsConstraints1 = wx.FlexGridSizer(0,5,2,2)
        bsConstraints2 = wx.FlexGridSizer(0,3,2,2)
        bsConstraints3 = wx.FlexGridSizer(0,1,2,2)
        for c in (bsConstraints1,bsConstraints2,bsConstraints3):
            self.bsConstraints.Add(c, 0, self.layoutFlags)
        
        self.edtFromTime = wx.TextCtrl(self.pnlConstraints, -1, "0")
        self.stFromTime = wx.StaticText(self.pnlConstraints, -1, "<=", style=wx.ALIGN_CENTER)
        self.stTime = wx.StaticText(self.pnlConstraints, -1, "Time", style=wx.ALIGN_CENTER)
        self.stToTime = wx.StaticText(self.pnlConstraints, -1, "<=", style=wx.ALIGN_CENTER)
        self.edtToTime = wx.TextCtrl(self.pnlConstraints, -1, "1")
        self.edtFromValue = wx.TextCtrl(self.pnlConstraints, -1, "0")
        self.stFromValue = wx.StaticText(self.pnlConstraints, -1, "<=", style=wx.ALIGN_CENTER)
        self.chConstrValue = wx.Choice(self.pnlConstraints, -1, choices=self.SplineOptim.GetNames())
        self.chConstrValue.SetSelection(0)
        self.stToValue = wx.StaticText(self.pnlConstraints, -1, "<=", style=wx.ALIGN_CENTER)
        self.edtToValue = wx.TextCtrl(self.pnlConstraints, -1, "1")

        self.edtFromTime.Bind(wx.EVT_KILL_FOCUS, self.OnConstraintOptions)
        self.edtToTime.Bind(wx.EVT_KILL_FOCUS, self.OnConstraintOptions)
        self.edtFromValue.Bind(wx.EVT_KILL_FOCUS, self.OnConstraintOptions)
        self.edtToValue.Bind(wx.EVT_KILL_FOCUS, self.OnConstraintOptions)
        self.chConstrValue.Bind(wx.EVT_CHOICE, self.OnConstraintOptions)
        for c in (self.edtFromTime,self.stFromTime,self.stTime,self.stToTime,self.edtToTime,
                  self.edtFromValue,self.stFromValue,self.chConstrValue,self.stToValue,self.edtToValue):
            c.SetMinSize((self.optionsSize/5.0-self.border*2.0, self.minHeight))
            bsConstraints1.Add(c, 0, self.layoutFlags, self.border)        

        self.btnAddConstraint = wx.Button(self.pnlConstraints, -1, "Add")
        self.btnRemoveConstraint = wx.Button(self.pnlConstraints, -1, "Remove")
        self.btnAddConstraint.Bind(wx.EVT_BUTTON, self.OnAddConstraint, id=self.btnAddConstraint.GetId())
        self.btnRemoveConstraint.Bind(wx.EVT_BUTTON, self.OnRemoveConstraint, id=self.btnRemoveConstraint.GetId())
        for c in (self.btnAddConstraint,self.btnRemoveConstraint):
            c.SetMinSize((self.optionsSize/2.0-self.border*2.0, self.minHeight))
            bsConstraints2.Add(c, 0, self.layoutFlags, self.border)        
        
        self.lbConstraints = wx.ListBox(self.pnlConstraints, -1)
        self.lbConstraints.Bind(wx.EVT_LISTBOX, self.OnConstraints)
        self.lbConstraints.SetMinSize((self.optionsSize-self.border*2.0, self.minHeight*6))
        bsConstraints3.Add(self.lbConstraints, 0, self.layoutFlags, self.border)        

        self.ShowConstraints()
        
        
    def SetupObjectives(self):
        bsObjectives1 = wx.FlexGridSizer(0,3,2,2)
        bsObjectives2 = wx.FlexGridSizer(0,3,2,2)
        bsObjectives3 = wx.FlexGridSizer(0,1,2,2)
        
        for c in (bsObjectives1,bsObjectives2,bsObjectives3):
            self.bsObjectives.Add(c, 0, self.layoutFlags, self.border)

        self.chObjNorm = wx.Choice(self.pnlObjectives, -1, choices=self.SplineOptim.GetNorms())
        self.chObjNorm.SetSelection(0)
        self.chObjValue = wx.Choice(self.pnlObjectives, -1, choices=self.SplineOptim.GetNames())
        self.chObjValue.SetSelection(0)
        self.edtObjWeight = wx.TextCtrl(self.pnlObjectives, -1, "1")
        self.chObjNorm.Bind(wx.EVT_CHOICE, self.OnObjectiveOptions)
        self.chObjValue.Bind(wx.EVT_CHOICE, self.OnObjectiveOptions)
        self.edtObjWeight.Bind(wx.EVT_KILL_FOCUS, self.OnObjectiveOptions)
        for c in (self.chObjNorm,self.chObjValue,self.edtObjWeight):        
            c.SetMinSize((self.optionsSize/3.0-2,self.minHeight))
            bsObjectives1.Add(c, 1, self.layoutFlags)

        self.btnAddObjective = wx.Button(self.pnlObjectives, -1, "Add")
        self.btnRemoveObjective = wx.Button(self.pnlObjectives, -1, "Remove")
        self.btnAddObjective.Bind(wx.EVT_BUTTON, self.OnAddObjective, id=self.btnAddObjective.GetId())
        self.btnRemoveObjective.Bind(wx.EVT_BUTTON, self.OnRemoveObjective, id=self.btnRemoveObjective.GetId())
        for c in (self.btnAddObjective,self.btnRemoveObjective):
            c.SetMinSize((self.optionsSize/2.0-self.border*2.0,self.minHeight))
            bsObjectives2.Add(c, 0, self.layoutFlags, self.border)
        
        self.lbObjectives = wx.ListBox(self.pnlObjectives, -1)
        self.lbObjectives.SetMinSize((self.optionsSize-self.border*2.0, self.minHeight*6))
        self.lbObjectives.Bind(wx.EVT_LISTBOX, self.OnObjectives)

        bsObjectives3.Add(self.lbObjectives, 0, self.layoutFlags, self.border)

        self.ShowObjectives()
        
    def PlotSolution(self):
        time = self.SplineOptim.GetTime()
        x0 = self.SplineOptim.GetSolution(0)
        x1 = self.SplineOptim.GetSolution(1) 
        x2 = self.SplineOptim.GetSolution(2) 
        data0 = vstack((time,x0))
        data1 = vstack((time,x1))
        data2 = vstack((time,x2))
        line0 = plot.PolyLine(data0.transpose(), colour='red', width=1)
        self.pcPosition.Draw(plot.PlotGraphics([line0], 'Position', 'time (s)', 'position (m)'))        
        line1 = plot.PolyLine(data1.transpose(), colour='red', width=1)
        self.pcVelocity.Draw(plot.PlotGraphics([line1], 'Velocity', 'time (s)', 'velocity (m/s)'))        
        line2 = plot.PolyLine(data2.transpose(), colour='red', width=1)
        self.pcAcceleration.Draw(plot.PlotGraphics([line2], 'Acceleration', 'time (s)', 'acceleration (m/s2)'))        
    
    def OnLoad(self, evt):
        wx.MessageBox("Load clicked", "Event handler", wx.OK)
                
    def OnSave(self, evt):
        wx.MessageBox("Save clicked", "Event handler", wx.OK)
                
    def OnQuit(self, evt):
        self.Close()

    def OnSolve(self, evt):
        retVal = self.SplineOptim.SolveProb()
        status = retVal[0]
        solTime = retVal[1]
        if status:
            self.btnSolve.SetForegroundColour(wx.Colour(0,180,0))
            self.statusBar.SetStatusText("Problem solved in " + str(solTime) + " s")
            self.PlotSolution()
            self.ShowConstraints()
        else:
            self.btnSolve.SetForegroundColour(wx.Colour(180,0,0))

    def OnAbout(self, evt):
        abtInfo = wx.AboutDialogInfo()
        abtInfo.SetName("SplineOptim")
        abtInfo.SetDescription("A program for motion law design (2018). "\
                               "Based on article: Optimal Splines for Rigid Motion Systems: A Convex Programming Framework, "\
                                "Bram Demeulenaere et al. Journal of Mechanical Design 131(10).")
        abtInfo.SetDevelopers(["Diederik Verscheure"])
        abtInfo.SetCopyright("GPL license")
        abtInfo.SetVersion("0.1")
        abtInfo.SetWebSite("https://github.com/diederikverscheure/splineoptim")
        wx.AboutBox(abtInfo)

    def OnSolveOptions(self, evt):
        units = self.chUnits.GetSelection()
        RPM = float(self.edtRPM.GetValue())
        ts = float(self.edtStartTime.GetValue())
        te = float(self.edtEndTime.GetValue())
        K = float(self.edtGridSize.GetValue())
        M = float(self.edtContLevel.GetValue()) 
        bctype = self.chBoundCondType.GetSelection()
        N = float(self.edtBoundCondLevel.GetValue())
        solver = self.chSolver.GetSelection()
        output = self.chOutputLevel.GetSelection()
        self.SplineOptim.SetOptions(units, RPM, ts, te, K, M, bctype, N, solver, output)
        self.ShowSolveOptions()
        self.ShowConstraints()
        self.ShowObjectives()
        
    def ShowSolveOptions(self):
        units = self.chUnits.GetSelection()
        if units == 0:
            self.edtRPM.Hide()
            self.stRPM.Hide()
        else:
            self.edtRPM.Show()
            self.stRPM.Show()
    
    def ShowConstraints(self):
        constraints = self.SplineOptim.GetConstraints()
        names = self.SplineOptim.GetNames()
        self.chConstrValue.SetItems(names)
        sel = self.lbConstraints.GetSelection()
        self.lbConstraints.Clear()
        for i in range(self.SplineOptim.GetNrConstraints()):
            k = self.SplineOptim.GetConstrValue(i)
            tl = self.SplineOptim.GetFromTime(i)
            tu = self.SplineOptim.GetToTime(i)
            xl = self.SplineOptim.GetFromValue(i)
            xu = self.SplineOptim.GetToValue(i)
            if self.SplineOptim.GetTotalConstrCost() > 0:
                c = round(self.SplineOptim.GetConstrCost(i)/self.SplineOptim.GetTotalConstrCost()*100*1000)/1000
            else:
                c = 0
            item = str(c) + "%" + (5-len(str(c)))*" " +  " - "
            if isfinite(tl) and isfinite(tu) and not (tl == tu): 
                item = item + str(tl) + " <= t <= " + str(tu)
            elif isfinite(tl) and isfinite(tu) and tl == tu: 
                item = item + "t == " + str(tl)
            elif isfinite(tl) and not isfinite(tu): 
                item = item + str(tl) + " <= t"
            elif not isfinite(tl) and isfinite(tu): 
                item = item + "t <= " + str(tu)
            else:
                item = item + "t"
            if isfinite(xl) and isfinite(xu) and not (xl == xu): 
                item = item + ", " + str(xl) + " <= " + names[k] + " <= " + str(xu)
            elif isfinite(xl) and isfinite(xu) and xl == xu: 
                item = item + ", " + names[k] + " == " + str(xl)
            elif isfinite(xl) and not isfinite(xu): 
                item = item + ", " + str(tl) + " <= " + names[k] 
            elif not isfinite(xl) and isfinite(xu): 
                item = item + ", " + names[k] + " <= " + str(xu) 
            else:
                item = item + ", " + names[k]
            self.lbConstraints.Append(item)
        if sel >= 0 and sel < self.SplineOptim.GetNrConstraints():
            self.lbConstraints.SetSelection(sel)
        else:
            if self.SplineOptim.GetNrConstraints() > 0:
                self.lbConstraints.SetSelection(0)
        self.OnConstraints(0)
            
    def OnAddConstraint(self, evt):
        k = self.chConstrValue.GetSelection()
        tl = float(self.edtFromTime.GetValue())
        tu = float(self.edtToTime.GetValue())
        xl = float(self.edtFromValue.GetValue())
        xu = float(self.edtToValue.GetValue())
        self.SplineOptim.AddConstraint(k, tl, tu, xl, xu)
        self.ShowConstraints()
        self.lbConstraints.SetSelection(self.SplineOptim.GetNrConstraints()-1)
        
    def OnRemoveConstraint(self, evt):
        i = self.lbConstraints.GetSelection()
        self.SplineOptim.RemoveConstraint(i)
        self.ShowConstraints()
        
    def OnConstraintOptions(self, evt):
        i = self.lbConstraints.GetSelection()
        if i >= 0:
            k = self.chConstrValue.GetSelection()
            try:
                tl = float(self.edtFromTime.GetValue())
            except ValueError:
                tl = -inf
            try:
                tu = float(self.edtToTime.GetValue())
            except ValueError:
                tu = inf
            try:
                xl = float(self.edtFromValue.GetValue())
            except ValueError:
                xl = -inf
            try:
                xu = float(self.edtToValue.GetValue())
            except ValueError:
                xu = inf
            self.SplineOptim.UpdateConstraint(i, k, tl, tu, xl, xu)
        self.ShowConstraints()
            
    def OnConstraints(self, evt):
        i = self.lbConstraints.GetSelection()
        if i >= 0:
            self.chConstrValue.SetSelection(self.SplineOptim.GetConstrValue(i))
            self.edtFromTime.SetValue(str(self.SplineOptim.GetFromTime(i)))
            self.edtToTime.SetValue(str(self.SplineOptim.GetToTime(i)))
            self.edtFromValue.SetValue(str(self.SplineOptim.GetFromValue(i)))
            self.edtToValue.SetValue(str(self.SplineOptim.GetToValue(i)))

    def ShowObjectives(self):
        objectives = self.SplineOptim.GetObjectives()
        names = self.SplineOptim.GetNames()
        self.chObjValue.SetItems(names)
        norms = self.SplineOptim.GetNorms()
        sel = self.lbObjectives.GetSelection()
        self.lbObjectives.Clear()
        for i in range(objectives.shape[0]):
            n = int(objectives[i,0])
            k = int(objectives[i,1])
            w = objectives[i,2]
            item = str(w) + " x " + norms[n] + " " + names[k]
            self.lbObjectives.Append(item)
        if sel >= 0 and sel < objectives.shape[0]:
            self.lbObjectives.SetSelection(sel)
        else:
            if objectives.shape[0] > 0:
                self.lbObjectives.SetSelection(0)
        self.OnObjectives(0)

    def OnAddObjective(self, evt):
        n = self.chObjNorm.GetSelection()
        k = self.chObjValue.GetSelection()
        w = float(self.edtObjWeight.GetValue())
        self.SplineOptim.AddObjective(n, k, w)
        objectives = self.SplineOptim.GetObjectives()
        self.ShowObjectives()
        self.lbObjectives.SetSelection(objectives.shape[0]-1)
        
    def OnRemoveObjective(self, evt):
        i = self.lbObjectives.GetSelection()
        self.SplineOptim.RemoveObjective(i)
        self.ShowObjectives()
        
    def OnObjectiveOptions(self, evt):
        i = self.lbObjectives.GetSelection()
        if i >= 0:
            n = self.chObjNorm.GetSelection()
            k = self.chObjValue.GetSelection()
            try:
                w = float(self.edtObjWeight.GetValue())
            except ValueError:
                w = 1
            self.SplineOptim.UpdateObjective(i, n, k, w)
        self.ShowObjectives()
            
    def OnObjectives(self, evt):
        objectives = self.SplineOptim.GetObjectives()
        i = self.lbObjectives.GetSelection()
        if i >= 0:
            self.chObjNorm.SetSelection(objectives[i,0])
            self.chObjValue.SetSelection(objectives[i,1])
            self.edtObjWeight.SetValue(str(objectives[i,2]))
        
class MainApp(wx.App):        
    def OnInit(self):
        frame = MainFrame()
        self.SetTopWindow(frame)
        frame.Show(True)
        return True
        
    def OnExit(self):
        return 0


def main(argv):
    app = MainApp()
    app.MainLoop()
    return 0

if __name__ == "__main__":
    main(sys.argv)
