import sys
import numpy as np
import cvxopt as cvx
from cvxopt import umfpack
from scipy.misc import factorial
import wx
import wx.adv
import time
import wx.lib.plot as plot
import wx.lib.scrolledpanel as scrolled
import xml.dom.minidom as minidom

class SplineOptim:
    def __init__(self):
        self.constraints = np.array([], dtype=float)
        self.objectives = np.array([], dtype=float)
        # units, RPM, start time, end time, grid size, cont level,
        # bound cond type, bound cond level, solver, output
        self.SetOptions('time', 300.0, 0.0, 10.0, 50, 2, 'zero', 2, 'glpk', 'verbose')
        self.AddConstraint(0, self.ts, self.ts, 1.0, 1.0)
        self.AddConstraint(0, self.te/2.0, self.te/2.0, 2.0, 2.0)
        self.AddConstraint(0, self.te, self.te, 1.5, 1.5)
        self.AddConstraint(0, self.ts, self.te, 1.0, 2.0)
        self.AddObjective(2, self.M+1, 1)
        self.AddObjective(0, self.M+2, 0.01)
        self.xs = np.zeros((self.K+1,self.M+3))

    def GetNames(self):
        names = ["position", "velocity", "acceleration", "jerk", "jolt"]
        for i in range(5,int(self.M)+2):
            name = str(i) + "th derivative"
            names = names + [name]
        return names[0:int(self.M)+3]
    
    def GetNorms(self):
        return ["one-norm", "two-norm", "inf-norm", "max. abs", "min. abs"]

    def GetUnits(self):
        return self.units        
    
    def GetUnitTypes(self):
        return ['time','degrees','radians']
        
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
    
    def GetBoundCondTypes(self):
        return ['zero','periodic','symmetric']

    def GetBoundCondLevel(self):
        return self.N

    def GetSolver(self):
        return self.solver
    
    def GetSolvers(self):
        return ['conelp','glpk','mosek','pulp']
    
    def GetOutput(self):
        return self.output
    
    def GetOutputTypes(self):
        return ['none','verbose']
                
    def GetTime(self):
        return np.linspace(self.ts,self.te,self.K+1)
    
    def GetSolution(self, k):
        if k <= self.xs.shape[1]:
            return self.xs[:,k]
            
    def GetConstraints(self):
        return self.constraints
    
    def SetConstraints(self,constraints):
        self.constraints = constraints

    def GetObjectives(self):
        return self.objectives
    
    def SetObjectives(self,objectives):
        self.objectives = objectives

    def GetDict(self):
        d = {'units': self.units, 
             'RPM': self.RPM,
             'ts': self.ts,
             'te': self.te,
             'K': self.K,
             'M': self.M,
             'bctype': self.bctype,
             'N': self.N,
             'solver': self.solver,
             'output': self.output,
             'constraints': self.constraints,
             'objectives': self.objectives
             }
        return d
    
    def SetDict(self,d):
        self.SetOptions(d['units'],d['RPM'],d['ts'],d['te'],d['K'],d['M'],d['bctype'],d['N'],d['solver'],d['output'])
        self.SetConstraints(d['constraints'])
        self.SetObjectives(d['objectives'])                

    def SetOptions(self, units, RPM, ts, te, K, M, bctype, N, solver, output):
        self.units = units
        self.RPM = RPM
        self.ts = ts
        self.te = te
        if self.units == 'time':
            self.RPM = 1.0/self.te*60
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
        if output == 'none':
            cvx.solvers.options['show_progress'] = False
            if self.solver == 'glpk':
                cvx.solvers.options['msg_lev'] = 'GLP_MSG_OFF'
        else:
            cvx.solvers.options['show_progress'] = True
            if self.solver == 'glpk':
                cvx.solvers.options['msg_lev'] = 'GLP_MSG_ON'

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
            j = np.setdiff1d(np.array(range(self.GetNrConstraints())),np.array([i]))
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
                self.constraints = np.array([[k, tl, tu, xl, xu, 0]], dtype=float)
            else:
                constraint = np.array([k, tl, tu, xl, xu, 0], dtype=float)
                self.constraints = np.vstack((self.constraints, constraint))

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
            j = np.setdiff1d(np.array(range(self.GetNrObjectives())),np.array([i]))
            self.objectives = self.objectives[j,:]

    def UpdateObjective(self, i, n, k, w):
        if i <= self.GetNrObjectives():
            self.SetObjNorm(i,n)
            self.SetObjValue(i,k)
            self.SetObjWeight(i,w)
        else:
            if self.GetNrObjectives() == 0:
                self.objectives = np.array([[n, k, w]], dtype=float)
            else:
                objective = np.array([n, k, w], dtype=float)
                self.objectives = np.vstack((self.objectives, objective))

    def SetupProb(self):
        # Integration conditions
        Ai = self.GetIntegration()
        bi = np.zeros((Ai.shape[0],1))
        # Boundary conditions
        Ab = self.GetBoundCond()
        bb = np.zeros((Ab.shape[0],1))
        # Equality conditions
        Aeq = np.vstack((Ai,Ab))
        beq = np.vstack((bi,bb))
        # Inequality conditions
        Aineq = np.zeros((0,(self.M+1)*(self.K+1)))
        bineq = np.zeros((Aineq.shape[0],1))
        # Slack variables?
        bs = np.zeros((0,1))
        c = np.zeros(((self.K+1)*(self.M+1),1))        
        t = self.GetTime()
        for i in range(self.GetNrObjectives()):
            n = self.GetObjNorm(i)
            k = self.GetObjValue(i)
            w = self.GetObjWeight(i)
            As = self.GetSlackCond(k)
            if n == 0:
                # one-norm
                c = np.vstack((c,w*np.ones((As.shape[0],1))/As.shape[0]))
                Az = np.zeros((As.shape[0],Aineq.shape[1]-(self.M+1)*(self.K+1)))
                Ao1 = np.hstack((-As,Az,-np.eye(As.shape[0])))
                Ao2 = np.hstack((As,Az,-np.eye(As.shape[0])))
                Ao = np.vstack((Ao1,Ao2))
            elif n == 1:
                # two-norm
                Ao = np.zeros((0,Aineq.shape[1]))
            elif n == 2:
                # inf-norm
                c = np.vstack((c,w))
                Az = np.zeros((As.shape[0],Aineq.shape[1]-(self.M+1)*(self.K+1)))
                Ao1 = np.hstack((-As,Az,-np.ones((As.shape[0],1))))
                Ao2 = np.hstack((As,Az,-np.ones((As.shape[0],1))))
                Ao = np.vstack((Ao1,Ao2))
            Aineq = np.vstack((np.hstack((Aineq,np.zeros((Aineq.shape[0],Ao.shape[1]-Aineq.shape[1])))),Ao))
            bineq = np.vstack((bineq,np.zeros((Ao.shape[0],1))))
        bs = np.vstack((bs,bineq.shape[0]))
        for i in range(self.GetNrConstraints()):
            k = self.GetConstrValue(i)
            tl = self.GetFromTime(i)
            tu = self.GetToTime(i)
            xl = self.GetFromValue(i)
            xu = self.GetToValue(i)
            ti = np.hstack((tl,t[(t>tl)*(t<tu)],tu))
            ti = np.unique(np.setdiff1d(ti,[-np.inf, np.inf]))
            Ac = np.zeros((ti.shape[0],(self.M+1)*(self.K+1)))
            idx = max(0,k)*(self.K+1) + np.interp(ti,t,range(t.shape[0]))
            for j in range(ti.shape[0]):
                if round(idx[j]) == idx[j]:
                    Ac[j,int(idx[j])] = 1
                else:
                    Ac[j,int(np.floor(idx[j]))] = (np.ceil(idx[j])-idx[j])
                    Ac[j,int(np.ceil(idx[j]))] = (idx[j]-np.floor(idx[j]))
            if np.isfinite(xl) and np.isfinite(xu) and xl == xu:
                Aeq = np.vstack((Aeq,Ac))
                beq = np.vstack((beq,xl*np.ones((ti.shape[0],1))))
            else:
                if np.isfinite(xl):
                    Aineq = np.vstack((Aineq,np.hstack((-Ac,np.zeros((Ac.shape[0],Aineq.shape[1]-Ac.shape[1]))))))
                    bineq = np.vstack((bineq,-xl*np.ones((ti.shape[0],1))))
                if np.isfinite(xu):
                    Aineq = np.vstack((Aineq,np.hstack((Ac,np.zeros((Ac.shape[0],Aineq.shape[1]-Ac.shape[1]))))))
                    bineq = np.vstack((bineq,xu*np.ones((ti.shape[0],1))))
            bs = np.vstack((bs,(xl != xu)*(np.isfinite(xl)*1+np.isfinite(xu)*1)*Ac.shape[0]))
        Aeq = np.hstack((Aeq,np.zeros((Aeq.shape[0],Aineq.shape[1]-Aeq.shape[1]))))

        self.presolve = False
        if self.presolve == True:
            # Presolve integration equations
            idx = np.hstack((range(0,self.M*(self.K+1),self.K+1),range(self.M*(self.K+1),(self.M+1)*(self.K+1))))
            didx = np.setdiff1d(range(0,(self.M+1)*(self.K+1)),idx)
            Ar = cvx.matrix(-Ai[:,idx])
            Al = cvx.sparse(cvx.matrix(Ai[:,didx]))
            umfpack.linsolve(Al,Ar)
            
            Ar = np.array(Ar)
            At = np.zeros((Ai.shape[1],Ar.shape[1]))
            At[didx,:] = Ar
            At[idx,:] = np.eye(Ar.shape[1])
            At = np.hstack((At,np.zeros((At.shape[0],Aeq.shape[1]-Ai.shape[1]))))
            At = np.vstack((At,np.hstack((np.zeros((Aeq.shape[1]-Ai.shape[1],Ar.shape[1])),np.eye(Aeq.shape[1]-Ai.shape[1])))))
            self.At = At
            
            Aeq = np.dot(Aeq,At)
            Aeq = Aeq[Ai.shape[0]:,:]
            beq = beq[Ai.shape[0]:,:]
            Aineq = np.dot(Aineq,At)
            c = np.dot(np.transpose(At),c)
        
        self.Aeq = cvx.sparse(cvx.matrix(Aeq))
        self.beq = cvx.matrix(beq)
        self.Aineq = cvx.sparse(cvx.matrix(Aineq))
        self.bineq = cvx.matrix(bineq)
        self.c = cvx.matrix(c)
        self.bs = bs
    
    def SolveProb(self):
        self.SetupProb()
        t1 = time.time()
        if self.solver == 'conelp':
            solver = None
        else:
            solver = self.solver
        sol = cvx.solvers.lp(self.c,self.Aineq,self.bineq,self.Aeq,self.beq,solver=solver)
        t2 = time.time()
        status = sol.get('status') == 'optimal'
        if status:
            xs = np.array(sol.get('x'))
            ys = np.array(sol.get('y'))
            ss = np.array(sol.get('s'))
            for i in range(self.GetNrConstraints()):
                if self.bs[i+1] != 0:
                    self.SetConstrCost(i, sum(ss[range(int(sum(self.bs[0:i+1])),int(sum(self.bs[0:i+1])+self.bs[i+1]))]))
            if self.presolve == True:
                xs = np.dot(self.At,xs)
            self.xs = np.reshape(xs[0:(self.M+1)*(self.K+1)],(self.K+1,self.M+1),'F')
            t = self.GetTime()
            xM1 = np.hstack((0,np.diff(self.xs[:,self.M],1,0)/np.diff(t),0))
            tm = np.hstack((t[0],t[0:-1]/2+t[1:]/2,t[-1]))
            xM1 = np.reshape(np.interp(t,tm,xM1),(-1,1))
            xM2 = np.reshape(np.hstack((0,np.diff(self.xs[:,self.M],2,0)/np.diff(t[0:-1])/np.diff(t[1:]),0)),(-1,1))
            self.xs = np.hstack((self.xs,xM1,xM2))
        return (status,t2-t1)
    
    def GetIntegration(self):
        dt = np.diff(self.GetTime())
        Ai = np.zeros((0,(self.M+1)*(self.K+1)))
        for n in range(1, self.M+1):
            An1 = np.hstack((np.diag(n*dt**n/factorial(n+1),0),np.zeros((self.K,1))))
            An1 = An1 + np.hstack((np.zeros((self.K,1)),np.diag(dt**n/factorial(n+1),0)))
            An2 = np.empty((self.K,0))
            for nv in range(1,n):
                An2 = np.hstack((An2,np.diag(dt**nv/factorial(nv),0),np.zeros((self.K,1))))
            An3 = np.hstack((np.eye(self.K),np.zeros((self.K,1))))
            An3 = An3 + np.hstack((np.zeros((self.K,1)),-np.eye(self.K)))            
            An4 = np.zeros((self.K,(self.M-n)*(self.K+1)))
            An = np.hstack((An4,An3,An2,An1))
            Ai = np.vstack((Ai,An))
        return Ai
    
    def GetBoundCond(self):
        if self.bctype == 'zero':
            # Zero
            Ab = np.zeros((2*self.N,(self.M+1)*(self.K+1)))
            for i in range(1,self.N+1):
                Ab[2*i-2,i*(self.K+1)] = 1
                Ab[2*i-1,(i+1)*(self.K+1)-1] = 1
        elif self.bctype == 'periodic':
            # Periodic
            Ab = np.zeros((self.N+1,(self.M+1)*(self.K+1)))
            for i in range(0,self.N+1):
                Ab[i,i*(self.K+1)] = 1
                Ab[i,(i+1)*(self.K+1)-1] = -1
        elif self.bctype == 'symmetric':
            # Symmetric
            Ab = np.zeros((self.N+1,(self.M+1)*(self.K+1)))
            for i in range(0,self.N+1):
                Ab[i,i*(self.K+1)] = 1
                Ab[i,(i+1)*(self.K+1)-1] = (-1.0)**(i+1)         
        return Ab

    def GetSlackCond(self,k):
        if k <= self.M:
            As = np.hstack((np.zeros((self.K+1,int(max(0,k))*(self.K+1))),np.eye(self.K+1),np.zeros((self.K+1,(self.M-int(max(0,k)))*(self.K+1)))))
        elif k == self.M+1:
            dt = np.diff(self.GetTime())
            Ap = np.hstack((np.diag(-1/dt,0),np.zeros((self.K,1))))
            Ap = Ap + np.hstack((np.zeros((self.K,1)),np.diag(1/dt,0)))
            As = np.hstack((np.zeros((self.K,self.M*(self.K+1))),Ap))
        elif k == self.M+2:
            dt = np.diff(self.GetTime())    
            App = np.hstack((np.diag(1/dt[0:-1]/dt[1:],0),np.zeros((self.K-1,2))))
            App = App + np.hstack((np.zeros((self.K-1,1)),-2*np.diag(1/dt[0:-1]/dt[1:],0),np.zeros((self.K-1,1))))
            App = App + np.hstack((np.zeros((self.K-1,2)),np.diag(1/dt[0:-1]/dt[1:],0)))
            As = np.hstack((np.zeros((self.K-1,self.M*(self.K+1))),App))
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
        self.ShowSolveOptions()
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
        self.pnlPlot.SetSizer(self.bsPlot)
        self.plotCanvas = []
        for i in range(0,self.SplineOptim.GetContLevel()+4):
            self.plotCanvas.insert(i,plot.PlotCanvas(self.pnlPlot))
            self.plotCanvas[i].SetInitialSize(self.plotSize)
            self.plotCanvas[i].enableGrid = True
            self.bsPlot.Add(self.plotCanvas[i], 0, self.layoutFlags, self.border)
        #self.plotCanvas[self.SplineOptim.GetContLevel()+3].SetEnableLegend(True)
        
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
        self.chUnits = wx.Choice(self.pnlSolveOptions, -1, choices=self.SplineOptim.GetUnitTypes())
        unitsSel = [i for i in range(0,len(self.SplineOptim.GetUnitTypes())) if self.SplineOptim.GetUnitTypes()[i] == self.SplineOptim.GetUnits()].pop(0)
        self.chUnits.SetSelection(unitsSel)

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
        self.chBoundCondType = wx.Choice(self.pnlSolveOptions, -1, choices=self.SplineOptim.GetBoundCondTypes())
        bctypeSel = [i for i in range(0,len(self.SplineOptim.GetBoundCondTypes())) if self.SplineOptim.GetBoundCondTypes()[i] == self.SplineOptim.GetBoundCondType()].pop(0)
        self.chBoundCondType.SetSelection(bctypeSel)

        self.stBoundCondLevel = wx.StaticText(self.pnlSolveOptions, -1, "Bound. level")
        self.edtBoundCondLevel = wx.TextCtrl(self.pnlSolveOptions, -1, str(self.SplineOptim.GetBoundCondLevel()))

        self.stSolver = wx.StaticText(self.pnlSolveOptions, -1, "Solver")
        self.chSolver = wx.Choice(self.pnlSolveOptions, -1, choices=self.SplineOptim.GetSolvers())
        solverSel = [i for i in range(0,len(self.SplineOptim.GetSolvers())) if self.SplineOptim.GetSolvers()[i] == self.SplineOptim.GetSolver()].pop(0)
        self.chSolver.SetSelection(solverSel)

        self.stOutput = wx.StaticText(self.pnlSolveOptions, -1, "Output level")        
        self.chOutput = wx.Choice(self.pnlSolveOptions, -1, choices=self.SplineOptim.GetOutputTypes())
        outputSel = [i for i in range(0,len(self.SplineOptim.GetOutputTypes())) if self.SplineOptim.GetOutputTypes()[i] == self.SplineOptim.GetOutput()].pop(0)
        self.chOutput.SetSelection(outputSel)
        
        self.chUnits.Bind(wx.EVT_CHOICE, self.OnSolveOptions)
        self.edtRPM.Bind(wx.EVT_KILL_FOCUS, self.OnSolveOptions)
        self.edtStartTime.Bind(wx.EVT_KILL_FOCUS, self.OnSolveOptions)
        self.edtEndTime.Bind(wx.EVT_KILL_FOCUS, self.OnSolveOptions)
        self.edtGridSize.Bind(wx.EVT_KILL_FOCUS, self.OnSolveOptions)
        self.edtContLevel.Bind(wx.EVT_KILL_FOCUS, self.OnSolveOptions)
        self.chBoundCondType.Bind(wx.EVT_CHOICE, self.OnSolveOptions)
        self.edtBoundCondLevel.Bind(wx.EVT_KILL_FOCUS, self.OnSolveOptions)
        self.chSolver.Bind(wx.EVT_CHOICE, self.OnSolveOptions)
        self.chOutput.Bind(wx.EVT_CHOICE, self.OnSolveOptions)
        
        for c in (self.btnSolve,self.btnLoad,self.btnSave):
            c.SetMinSize((self.optionsSize/3.0-self.border*2.0, self.minHeight))
            bsSolveOptions1.Add(c, 0, self.layoutFlags, self.border)
        
        for c in (self.stUnits,self.chUnits,self.stRPM,self.edtRPM,
                  self.stStartTime,self.edtStartTime,self.stEndTime,self.edtEndTime,
                  self.stGridSize,self.edtGridSize,self.stContLevel,self.edtContLevel,
                  self.stBoundCondType,self.chBoundCondType,self.stBoundCondLevel,self.edtBoundCondLevel,
                  self.stSolver,self.chSolver,self.stOutput,self.chOutput):
            c.SetMinSize((self.optionsSize/4.0-self.border*2.0, self.minHeight))
            bsSolveOptions2.Add(c, 0, self.layoutFlags, self.border)
                
    def SetupConstraints(self):
        bsConstraints1 = wx.FlexGridSizer(0,5,2,2)
        bsConstraints2 = wx.FlexGridSizer(0,3,2,2)
        bsConstraints3 = wx.FlexGridSizer(0,1,2,2)
        for c in (bsConstraints1,bsConstraints2,bsConstraints3):
            self.bsConstraints.Add(c, 0, self.layoutFlags)
        
        self.edtFromTime = wx.TextCtrl(self.pnlConstraints, -1, "0")
        self.stFromTime = wx.StaticText(self.pnlConstraints, -1, "<=", style=wx.ALIGN_CENTER)
        self.stTime = wx.StaticText(self.pnlConstraints, -1, "time", style=wx.ALIGN_CENTER)
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
        names = self.SplineOptim.GetNames()
        norms = self.SplineOptim.GetNorms()
        constraints = self.SplineOptim.GetConstraints()
        colour = ['blue','green','black','violet']
        for i in range(0,self.SplineOptim.GetContLevel()+3):
            lines = []
            x = self.SplineOptim.GetSolution(i)
            data = np.vstack((time,x))
            line = plot.PolyLine(data.transpose(), colour='blue', width=1, legend=names[i].lower())
            lines.insert(len(lines),line)
            for j in range(0,constraints.shape[0]):
                if int(constraints[j,0]) == i:
                    tlu = np.array([constraints[j,1],constraints[j,2],constraints[j,2],constraints[j,1],constraints[j,1]])
                    xlu = np.array([constraints[j,3],constraints[j,3],constraints[j,4],constraints[j,4],constraints[j,3]])
                    datalu = np.vstack((tlu,xlu))
                    if constraints[j,1] == constraints[j,2]:
                        markerlu = plot.PolyMarker(datalu.transpose(), marker='cross', colour='red', width=1, legend='constraint '+str(j+1))
                        lines.insert(len(lines),markerlu)
                    else:
                        linelu = plot.PolyLine(datalu.transpose(), colour='red', width=1, legend='constraint '+str(j+1))
                        lines.insert(len(lines),linelu)
            xAxis = 'time [s]'
            yAxis = names[i].lower() + ' (m/s^' + str(i) + ')'
            self.plotCanvas[i].Draw(plot.PlotGraphics(lines, names[i], xAxis, yAxis))        
            self.plotCanvas[i].enableLegend = True
            self.plotCanvas[i].fontSizeLegend = 8

        line = []
        for i in range(0,self.SplineOptim.GetNrObjectives()):
            wi = self.SplineOptim.GetObjWeight(i)
            ni = int(self.SplineOptim.GetObjNorm(i))
            vi = int(self.SplineOptim.GetObjValue(i))
            if ni == 0:
                x = wi*abs(self.SplineOptim.GetSolution(vi))
            elif ni == 1:
                x = wi*self.SplineOptim.GetSolution(vi)**2.0
            elif ni >= 2:
                x = wi*abs(self.SplineOptim.GetSolution(vi))
            data = np.vstack((time,x))
            line.insert(i,plot.PolyLine(data.transpose(), colour=colour[i], width=1, legend=norms[ni] + ' ' + names[vi]))
        xAxis = 'time [s]'
        yAxis = 'weighted objective functions [-]'
        self.plotCanvas[self.SplineOptim.GetContLevel()+3].Draw(plot.PlotGraphics(line, 'Objective functions', xAxis, yAxis))        
        self.plotCanvas[self.SplineOptim.GetContLevel()+3].enableLegend = True
        self.plotCanvas[self.SplineOptim.GetContLevel()+3].fontSizeLegend = 8

    def OnLoad(self, evt):
        flDialog = wx.FileDialog(self, defaultFile='splineoptim.xml', wildcard='*.xml', style=wx.FD_OPEN)
        if flDialog.ShowModal() == wx.ID_CANCEL:
            return
        fileName = flDialog.GetPath()
        doc = minidom.parse(fileName)
        d = dict()
        for child in doc.childNodes:
            for child2 in child.childNodes:
                if child2.hasChildNodes():
                    if child2.childNodes[0].nodeValue.isalpha():
                        d[child2.nodeName] = child2.childNodes[0].nodeValue
                    else:
                        d[child2.nodeName] = eval(child2.childNodes[0].nodeValue)
        d['constraints'] = np.array(d['constraints'])
        d['objectives'] = np.array(d['objectives'])
        self.SplineOptim.SetDict(d)
        self.ShowSolveOptions()
        self.ShowConstraints()
        self.ShowObjectives()
                
    def OnSave(self, evt):
        flDialog = wx.FileDialog(self, defaultFile='splineoptim.xml', wildcard='*.xml', style=wx.FD_SAVE)
        if flDialog.ShowModal() == wx.ID_CANCEL:
            return
        fileName = flDialog.GetPath()
        doc = minidom.Document()
        root = doc.createElement('SplineOptim')
        XMLvalues = self.SplineOptim.GetDict()
        for value in XMLvalues:
            tempChild = doc.createElement(value)
            root.appendChild(tempChild)
            if (type(XMLvalues[value]) == int) or(type(XMLvalues[value]) == float) or (type(XMLvalues[value]) == str):
                nodeText = doc.createTextNode(str(XMLvalues[value]).strip())
            else:
                nodeText = doc.createTextNode(np.array2string(XMLvalues[value],separator=',').strip())
            tempChild.appendChild(nodeText)
        doc.appendChild(root)
        doc.writexml(open(fileName,'w'),indent=' ',addindent=' ',newl='\n')
        doc.unlink()        
                
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
        title = 'SplineOptim'
        text = "A program for motion law design (2018). "\
               "Based on article: Optimal Splines for Rigid Motion Systems: A Convex Programming Framework, "\
               "B. Demeulenaere, J. De Caigny, G. Pipeleers, J. De Schutter and J. Swevers, "\
               "Journal of Mechanical Design 131(10)."
        developer = 'Diederik Verscheure'
        abtInfo = wx.adv.AboutDialogInfo()
        abtInfo.SetName(title)
        abtInfo.SetDescription(text)
        abtInfo.SetDevelopers([developer])
        abtInfo.SetCopyright("GPL license")
        abtInfo.SetVersion("0.1")
        abtInfo.SetWebSite("https://github.com/diederikverscheure/splineoptim")
        wx.adv.AboutBox(abtInfo)

    def OnSolveOptions(self, evt):
        units = self.SplineOptim.GetUnitTypes()[self.chUnits.GetSelection()]
        RPM = float(self.edtRPM.GetValue())
        if self.SplineOptim.GetUnits() == 'time':
            ts = float(self.edtStartTime.GetValue())
            te = float(self.edtEndTime.GetValue())
        else:
            ts = float(self.edtStartTime.GetValue())/360.0*60.0/RPM
            te = float(self.edtEndTime.GetValue())/360.0*60.0/RPM            
        K = float(self.edtGridSize.GetValue())
        M = float(self.edtContLevel.GetValue()) 
        bctype = self.SplineOptim.GetBoundCondTypes()[self.chBoundCondType.GetSelection()]
        N = float(self.edtBoundCondLevel.GetValue())
        solver = self.SplineOptim.GetSolvers()[self.chSolver.GetSelection()]
        output = self.SplineOptim.GetOutputTypes()[self.chOutput.GetSelection()]
        self.SplineOptim.SetOptions(units, RPM, ts, te, K, M, bctype, N, solver, output)
        self.ShowSolveOptions()
        self.ShowConstraints()
        self.ShowObjectives()
        
    def ShowSolveOptions(self):
        units = self.SplineOptim.GetUnits()
        unitsSel = [i for i in range(0,len(self.SplineOptim.GetUnitTypes())) if self.SplineOptim.GetUnitTypes()[i] == units].pop(0)
        self.chUnits.SetSelection(unitsSel)
        RPM = self.SplineOptim.GetRPM()
        self.edtRPM.SetValue(str(RPM))
        ts = self.SplineOptim.GetStartTime()
        self.edtStartTime.SetValue(str(ts))
        te = self.SplineOptim.GetEndTime()
        self.edtEndTime.SetValue(str(te))
        K = self.SplineOptim.GetGridSize()
        self.edtGridSize.SetValue(str(K))
        M = self.SplineOptim.GetContLevel()
        self.edtContLevel.SetValue(str(M)) 
        bctype = self.SplineOptim.GetBoundCondType()
        bctypeSel = [i for i in range(0,len(self.SplineOptim.GetBoundCondTypes())) if self.SplineOptim.GetBoundCondTypes()[i] == bctype].pop(0)
        self.chBoundCondType.SetSelection(bctypeSel)
        N = self.SplineOptim.GetBoundCondLevel()
        self.edtBoundCondLevel.SetValue(str(N))
        solver = self.SplineOptim.GetSolver()
        solverSel = [i for i in range(0,len(self.SplineOptim.GetSolvers())) if self.SplineOptim.GetSolvers()[i] == solver].pop(0)
        self.chSolver.SetSelection(solverSel)
        output = self.SplineOptim.GetOutput()
        outputSel = [i for i in range(0,len(self.SplineOptim.GetOutputTypes())) if self.SplineOptim.GetOutputTypes()[i] == output].pop(0)
        self.chOutput.SetSelection(outputSel)        
        if units == 'time':
            self.edtRPM.Hide()
            self.stRPM.Hide()
            self.stStartTime.SetLabel('Start time')
            self.stEndTime.SetLabel('End time')
            self.stTime.SetLabel('time')
        else:
            self.edtRPM.Show()
            self.stRPM.Show()
            self.stStartTime.SetLabel('Start angle')
            self.stEndTime.SetLabel('End angle')
            self.stTime.SetLabel('angle')
    
    def ShowConstraints(self):
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
            item = str(i+1) + ') ' + str(c) + "%" + (5-len(str(c)))*" " +  " - "
            if np.isfinite(tl) and np.isfinite(tu) and not (tl == tu): 
                item = item + str(tl) + " <= t <= " + str(tu)
            elif np.isfinite(tl) and np.isfinite(tu) and tl == tu: 
                item = item + "t == " + str(tl)
            elif np.isfinite(tl) and not np.isfinite(tu): 
                item = item + str(tl) + " <= t"
            elif not np.isfinite(tl) and np.isfinite(tu): 
                item = item + "t <= " + str(tu)
            else:
                item = item + "t"
            if np.isfinite(xl) and np.isfinite(xu) and not (xl == xu): 
                item = item + ", " + str(xl) + " <= " + names[k] + " <= " + str(xu)
            elif np.isfinite(xl) and np.isfinite(xu) and xl == xu: 
                item = item + ", " + names[k] + " == " + str(xl)
            elif np.isfinite(xl) and not np.isfinite(xu): 
                item = item + ", " + str(tl) + " <= " + names[k] 
            elif not np.isfinite(xl) and np.isfinite(xu): 
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
                tl = -np.inf
            try:
                tu = float(self.edtToTime.GetValue())
            except ValueError:
                tu = np.inf
            try:
                xl = float(self.edtFromValue.GetValue())
            except ValueError:
                xl = -np.inf
            try:
                xu = float(self.edtToValue.GetValue())
            except ValueError:
                xu = np.inf
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
            item = str(i+1) + ') ' + str(w) + " x " + norms[n] + " " + names[k]
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
