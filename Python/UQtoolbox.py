#UQtoolbox
#Module for calculating local sensitivity indices, parameter correlation, and Sobol indices for an arbitrary model
#Authors: Harley Hanes, NCSU, hhanes@ncsu.edu
#Required Modules: numpy, seaborne
#Functions: LSA->GetJacobian
#           GSA->ParamSample, PlotGSA, GetSobol

import numpy as np
import sys
import warnings
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from tabulate import tabulate                       #Used for printing tables to terminal
#import sobol                                        #Used for generating sobol sequences
import SALib.sample as sample
import scipy.stats as sct
#import seaborne as seaborne
###----------------------------------------------------------------------------------------------
###-------------------------------------Class Definitions----------------------------------------
###----------------------------------------------------------------------------------------------

##--------------------------------------uqOptions--------------------------------------------------
#Define class "uqOptions", this will be the class used to collect algorithm options for functions
#   -Subclasses: lsaOptions, plotOptions, gsaOptions
#--------------------------------------lsaOptions------------------------------------------------
class lsaOptions:
    def __init__(self,run=True, runActiveSubspace=True, xDelta=10**(-12),\
                 method='complex', scale='y', subspaceRelTol=.001):
        self.run=run                              #Whether to run lsa (True or False)
        self.xDelta=xDelta                        #Input perturbation for calculating jacobian
        self.scale=scale                          #scale can be y, n, or both for outputing scaled, unscaled, or both
        self.method=method                        #method used for approximating derivatives
        if self.run == False:
            self.runActiveSubspace = False
        else:
            self.runActiveSubspace=runActiveSubspace
        self.subspaceRelTol=subspaceRelTol
        if not self.scale.lower() in ('y','n','both'):
            raise Exception('Error! Unrecgonized scaling output, please enter y, n, or both')
        if not self.method.lower() in ('complex','finite'):
            raise Exception('Error! unrecognized derivative approx method. Use complex or finite')
        if self.xDelta<0 or not isinstance(self.xDelta,float):
            raise Exception('Error! Non-compatibale xDelta, please use a positive floating point number')
        if self.subspaceRelTol<0 or self.subspaceRelTol>1 or not isinstance(self.xDelta,float):
            raise Exception('Error! Non-compatibale xDelta, please use a positive floating point number less than 1')
    pass
#--------------------------------------gsaOptions------------------------------------------------
class gsaOptions:
    def __init__(self, run = True, runSobol=True, runMorris=True, nSampSobol=100000, \
                 nSampMorris=4, lMorris=3):
        self.run = run
        if self.run == False:
            self.runSobol = False
            self.runMorris = False
        else:
            self.runSobol=runSobol                            #Whether to run Sobol (True or False)
            self.runMorris=runMorris                          #Whether to run Morris (True or False)
        self.nSampSobol = nSampSobol                      #Number of samples to be generated for GSA
        self.nSampMorris = nSampMorris
        self.lMorris=lMorris
        pass
#--------------------------------------plotOptions------------------------------------------------
class plotOptions:
    def __init__(self,run=True,nPoints=400,path=False):
        self.run=run
        self.nPoints=nPoints
        self.path=path
        pass
#--------------------------------------uqOptions------------------------------------------------
#   Class holding the above options subclasses
class uqOptions:
    def __init__(self,lsa=lsaOptions(),plot=plotOptions(),gsa=gsaOptions(), \
                 display=True, save=False, path='..'):
        self.lsa=lsa
        self.plot=plot
        self.gsa=gsa
        self.display=display                       #Whether to print results to terminal
        self.save=save                             #Whether to save results to files
        self.path=path                             #Where to save files
        if self.save and not self.path:
            warnings.warn("Save marked as true but no path given, saving files to current folder.")
            path=''
    pass

##-------------------------------------model------------------------------------------------------
#Define class "model", this will be the class used to collect input information for all functions
class model:
    #Model sets should be initialized with base parameter settings, covariance Matrix, and eval function that
    #   takes in a vector of POIs and outputs a vector of QOIs
    def __init__(self,basePOIs=np.empty(0), POInames = np.empty(0), \
                 QOInames= np. empty(0), cov=np.empty(0), \
                 evalFcn=np.empty(0), dist='unif',distParms='null'):
        self.basePOIs=basePOIs
        if not isinstance(self.basePOIs,np.ndarray):                    #Confirm that basePOIs is a numpy array
            warnings.warn("model.basePOIs is not a numpy array")
        if np.ndim(self.basePOIs)>1:                                    #Check to see if basePOIs is a vector
            self.basePOIs=np.squeeze(self.basePOIs)                     #Make a vector if an array with 1 dim greater than 1
            if np.ndim(self.basePOIs)!=1:                               #Issue an error if basePOIs is a matrix or tensor
                raise Exception("Error! More than one dimension of size 1 detected for model.basePOIs, model.basePOIs must be dimension 1")
            else:                                                       #Issue a warning if dimensions were squeezed out of base POIs
                warnings.warn("model.basePOIs was reduced a dimension 1 array. No entries were deleted.")
        self.nPOIs=self.basePOIs.size
        #Assign POInames
        self.POInames = POInames                                            #Assign POInames called
        if (self.POInames.size != self.nPOIs) & (self.POInames.size !=0):   #Check that correct size if given
            warnings.warn("POInames entered but the number of names does not match the number of POIs. Ignoring names.")
            self.POInames=np.empty(0)
        if self.POInames.size==0:                                           #If not given or incorrect size, number POIs
            POInumbers=np.arange(0,self.nPOIs)
            self.POInames=np.char.add('POI',POInumbers.astype('U'))
        #Assign evaluation function and compute baseQOIs
        self.evalFcn=evalFcn
        self.baseQOIs=evalFcn(basePOIs)
        if not isinstance(self.baseQOIs,np.ndarray):                    #Confirm that baseQOIs is a numpy array
            warnings.warn("model.baseQOIs is not a numpy array")
        print(self.baseQOIs)
        self.nQOIs=len(self.baseQOIs)
        #Assign QOI names
        self.QOInames = QOInames
        if (self.QOInames.size !=self.nQOIs) & (self.QOInames.size !=0):    #Check names if given match number of QOIs
            warnings.warn("QOInames entered but the number of names does not match the number of QOIs. Ignoring names.")
            self.QOInames = np.empty(0)
        if self.QOInames.size==0:                                 #If not given or incorrect size, number QOIs
            QOInumbers = np.arange(0, self.nQOIs)
            self.QOInames = np.char.add('QOI', QOInumbers.astype('U'))
        #Assign covariance matrix
        self.cov=cov
        if self.cov.size!=0 and np.shape(self.cov)!=(self.nPOIs,self.nPOIs): #Check correct sizing
            raise Exception("Error! model.cov is not an nPOI x nPOI array")
        #Assign distributions
        self.dist = dist                        #String identifying sampling distribution for parameters
                                                #       Supported distributions: unif, normal, exponential, beta, inverseCDF
        if isinstance(distParms,str):
            if self.dist.lower()=='uniform':
                self.distParms=[[.8],[1.2]]*np.ones((2,self.nPOIs))*self.basePOIs
            elif self.dist.lower()=='normal':
                if cov.size()==0:
                    self.distParms=[[1],[.2]]*np.ones((2,self.nPOIs))*self.basePOIs
                else:
                    self.distParms=[self.basePOIs, np.diag(self.cov,k=0)]
            elif distParms.lower() == 'null':
                self.distParms = distParms
            else:
                raise Exception("Unrecognized entry for distParms: " + str(distParms))

        else:
            self.distParms=distParms
    pass
    def copy(self):
        return model(basePOIs=self.basePOIs, POInames = self.POInames, QOInames= self.QOInames, cov=self.cov, \
                 evalFcn=self.evalFcn, dist=self.dist,distParms=self.distParms)

##------------------------------------results-----------------------------------------------------
#-------------------------------------lsaResults--------------------------------------------------
# Define class "lsa", this will be the used to collect relative sensitivity analysis outputs
class lsaResults:
    #
    def __init__(self,jacobian=np.empty, rsi=np.empty, fisher=np.empty, reducedModel=np.empty, activeSpace="", inactiveSpace=""):
        self.jac=jacobian
        self.rsi=rsi
        self.fisher=fisher
        self.reducedModel=reducedModel
        self.activeSpace=activeSpace
        self.inactiveSpace=inactiveSpace
    pass
#-------------------------------------gsaResults--------------------------------------------------
# Define class "gsaResults" which holds sobol analysis results
class gsaResults:
    #
    def __init__(self,sobolBase=np.empty, sobolTot=np.empty, fA=np.empty, fB=np.empty, fD=np.empty, fAB=np.empty, \
                 sampD=np.empty,sigma2=np.empty, muStar=np.empty):
        self.sobolBase=sobolBase
        self.sobolTot=sobolTot
        self.fA=fA
        self.fB=fB
        self.fD=fD
        self.fAB=fAB
        self.sampD=sampD
        self.muStar=muStar
        self.sigma2=sigma2
    pass
##------------------------------------results-----------------------------------------------------
# Define class "results" which holds a gsaResults object and lsaResults object

class results:
    def __init__(self,lsa=lsaResults(), gsa=gsaResults()):
        self.lsa=lsa
        self.gsa=gsa
    pass


###----------------------------------------------------------------------------------------------
###-------------------------------------Main Functions----------------------------------------
###----------------------------------------------------------------------------------------------
#   The following functions are the primary functions for running the package. RunUQ runs both local sensitivity
#   analysis and global sensitivity analysis while printing to command window summary statistics. However, local
#   sensitivity analysis and global sensitivity analysis can be run independently with LSA and GSA respectively

##--------------------------------------RunUQ-----------------------------------------------------
def RunUQ(model, options):
    #RunUQ is the primary call function for UQtoolbox and runs both the local sensitivity analysis and global sensitivity
    #   analysis while printing summary statistics to the command window.
    #Inputs: model object, options object
    #Outpts: results object, a list of summary results is printed to command window

    #Run Local Sensitivity Analysis
    if options.lsa.run:
        results.lsa = LSA(model, options)

    #Run Global Sensitivity Analysis
    # if options.gsa.run:
        # if options.lsa.run:
        #     #Use a reduced model if it was caluclated
        #     results.gsa=GSA(results.lsa.reducedModel, options)
        # else:
    if options.gsa.run:
        results.gsa = GSA(model, options)

    #Print Results
    if options.display:
        PrintResults(results,model,options)                     #Print results to standard output path

    if options.save:
        original_stdout = sys.stdout                            #Save normal output path
        sys.stdout=open(options.path + 'Results.txt', 'a+')            #Change output path to results file
        #PrintResults(results,model,options)                     #Print results to file
        sys.stdout=original_stdout                              #Revert normal output path
        SaveResults(results,model,options)

    #Plot Samples
    if options.gsa.runSobol & options.gsa.run:
        PlotGSA(model, results.gsa.sampD, results.gsa.fD, options)

    return results

# Top order functions- These functions are the main functions for each component of our analysis they include,

##--------------------------------------LSA-----------------------------------------------------
# Local Sensitivity Analysis main
def LSA(model, options):
    # LSA implements the following local sensitivity analysis methods on system specified by "model" object
        # 1) Jacobian
        # 2) Scaled Jacobian for Relative Sensitivity Index (RSI)
        # 3) Fisher Information matrix
    # Required Inputs: object of class "model" and object of class "options"
    # Outputs: Object of class lsa with Jacobian, RSI, and Fisher information matrix

    # Calculate Jacobian
    jacRaw=GetJacobian(model.evalFcn, model.basePOIs, options.lsa, scale=False, yBase=model.baseQOIs)
    # Calculate relative sensitivity index (RSI)
    jacRSI=GetJacobian(model.evalFcn, model.basePOIs, options.lsa, scale=True, yBase=model.baseQOIs)
    # Calculate Fisher Information Matrix from jacobian
    fisherMat=np.dot(np.transpose(jacRaw), jacRaw)

    #Active Subspace Analysis
    if options.lsa.runActiveSubspace:
        reducedModel, activeSpace, inactiveSpace = GetActiveSubspace(model, options.lsa)
        #Collect Outputs and return as an lsa object
        return lsaResults(jacobian=jacRaw, rsi=jacRSI, fisher=fisherMat,\
                          reducedModel=reducedModel, activeSpace=activeSpace,\
                          inactiveSpace=inactiveSpace)
    else:
        return lsaResults(jacobian=jacRaw, rsi=jacRSI, fisher=fisherMat)

    



##--------------------------------------GSA-----------------------------------------------------
def GSA(model, options):
    #GSA implements the following local sensitivity analysis methods on "model" object
        # 1) Gets sampling distribution (used only for internal calculations)
        # 2) Calculates Sobol Indices
        # 3) Performs Morris Screenings (not yet implemented)
        # 4) Produces histogram plots for QOI values (not yet implemented)
    # Required Inputs: Object of class "model" and object of class "options"
    # Outputs: Object of class gsa with fisher and sobol elements

    #Get Parameter Distributions
    model=GetSampDist(model, options.gsa)

    #Morris Screening
    if options.gsa.runMorris:
        muStar, sigma2 = CaclulateMorris(model, options)
        results=gsaResults(muStar=muStar, sigma2=sigma2)

    #Sobol Analysis
    if options.gsa.runSobol:
        #Make Distribution Samples and Calculate model results
        [fA, fB, fAB, fD, sampD] = GetSamples(model, options.gsa)
        #Calculate Sobol Indices
        [sobolBase, sobolTot]=CalculateSobol(fA, fB, fAB, fD)
        if options.gsa.runMorris:
            results.fD=fD
            results.fA=fA
            results.fB=fB
            results.fAB=fAB
            results.sampD=sampD
            results.sobolBase=sobolBase
            results.sobolTot=sobolTot
        else:
            results=gsaResults(fD=fD, fA=fA, fB=fB, fAB=fAB, sampD= sampD, sobolBase=sobolBase, sobolTot=sobolTot)

    return results

def PrintResults(results,model,options):
    # Print Results
    #Results Header
    print('Sensitivity results for nSampSobol=' + str(options.gsa.nSampSobol))
    #Local Sensitivity Analysis
    if options.lsa.run:
        print('\n Base POI Values')
        print(tabulate([model.basePOIs], headers=model.POInames))
        print('\n Base QOI Values')
        print(tabulate([model.baseQOIs], headers=model.QOInames))
        print('\n Sensitivity Indices')
        print(tabulate(np.concatenate((model.POInames.reshape(model.nPOIs,1),np.transpose(results.lsa.jac)),1),
              headers= np.append("",model.QOInames)))
        print('\n Relative Sensitivity Indices')
        print(tabulate(np.concatenate((model.POInames.reshape(model.nPOIs,1),np.transpose(results.lsa.rsi)),1),
              headers= np.append("",model.QOInames)))
        #print("Fisher Matrix: " + str(results.lsa.fisher))
        #Active Subsapce Analysis
        print('\n Active Supspace')
        print(results.lsa.activeSpace)
        print('\n Inactive Supspace')
        print(results.lsa.inactiveSpace)
    if options.gsa.run: 
        if options.gsa.runSobol:
            if model.nQOIs==1:
                print('\n Sobol Indices for ' + model.QOInames[0])
                print(tabulate(np.concatenate((model.POInames.reshape(model.nPOIs,1), results.gsa.sobolBase.reshape(model.nPOIs,1), \
                                               results.gsa.sobolTot.reshape(model.nPOIs,1)), 1),
                               headers=["", "1st Order", "Total Sensitivity"]))
            else:
                for iQOI in range(0,model.nQOIs):
                    print('\n Sobol Indices for '+ model.QOInames[iQOI])
                    print(tabulate(np.concatenate((model.POInames.reshape(model.nPOIs,1),results.gsa.sobolBase[[iQOI],:].reshape(model.nPOIs,1), \
                        results.gsa.sobolTot[[iQOI],:].reshape(model.nPOIs,1)),1), headers = ["", "1st Order", "Total Sensitivity"]))
    
        if options.gsa.runMorris:
            if model.nQOIs==1:
                print('\n Morris Screening Results for' + model.QOInames[0])
                print(tabulate(np.concatenate((model.POInames.reshape(model.nPOIs, 1), results.gsa.muStar.reshape(model.nPOIs, 1), \
                                               results.gsa.sigma2.reshape(model.nPOIs, 1)), 1),
                    headers=["", "muStar", "sigma2"]))
            else:
                print('\n Morris Screening Results for' + model.QOInames[iQOI])
                print(tabulate(np.concatenate(
                    (model.POInames.reshape(model.nPOIs, 1), results.gsa.muStar[[iQOI], :].reshape(model.nPOIs, 1), \
                     results.gsa.sigma2[[iQOI], :].reshape(model.nPOIs, 1)), 1),
                    headers=["", "muStar", "sigma2"]))
def SaveResults(results,model,options):
    np.savez(options.path + 'results_lsa.npz', basePOIs = model.basePOIs, baseQOIs = model.baseQOIs, \
             sensitivities = results.lsa.jac, rel_sensitivities = results.lsa.rsi)
    

###----------------------------------------------------------------------------------------------
###-------------------------------------Support Functions----------------------------------------
###----------------------------------------------------------------------------------------------

##--------------------------------------GetJacobian-----------------------------------------------------
def GetJacobian(evalFcn, xBase, lsaOptions, **kwargs):
    # GetJacobian calculates the Jacobian for n QOIs and p POIs
    # Required Inputs: object of class "model" (.cov element not required)
    #                  object of class "lsaOptions"
    # Optional Inputs: alternate POI position to estimate Jacobian at (*arg) or complex step size (h)
    if 'scale' in kwargs:                                                   # Determine whether to scale derivatives
                                                                            #   (for use in relative sensitivity indices)
        scale = kwargs["scale"]
        if not isinstance(scale, bool):                                     # Check scale value is boolean
            raise Exception("Non-boolean value provided for 'scale' ")      # Stop compiling if not
    else:
        scale = False                                                       # Function defaults to no scaling
    if 'yBase' in kwargs:
        yBase = kwargs["yBase"]
    else:
        yBase = evalFcn(xBase)

    #Load options parameters for increased readibility
    xDelta=lsaOptions.xDelta

    #Initialize base QOI value, the number of POIs, and number of QOIs
    nPOIs = np.size(xBase)
    nQOIs = np.size(yBase)

    jac = np.empty(shape=(nQOIs, nPOIs), dtype=float)                       # Define Empty Jacobian Matrix

    for iPOI in range(0, nPOIs):                                            # Loop through POIs
        # Isolate Parameters
        if lsaOptions.method.lower()== 'complex':
            xPert = xBase + np.zeros(shape=xBase.shape)*1j                  # Initialize Complex Perturbed input value
            xPert[iPOI] += xDelta * 1j                                      # Add complex Step in input
        elif lsaOptions.method.lower() == 'finite':
            xPert=xBase*(1+xDelta)
        yPert = evalFcn(xPert)                                        # Calculate perturbed output
        for jQOI in range(0, nQOIs):                                        # Loop through QOIs
            if lsaOptions.method.lower()== 'complex':
                jac[jQOI, iPOI] = np.imag(yPert[jQOI] / xDelta)                 # Estimate Derivative w/ 2nd order complex
            elif lsaOptions.method.lower() == 'finite':
                jac[jQOI, iPOI] = (yPert[jQOI]-yBase[jQOI]) / xDelta
            #Only Scale Jacobian if 'scale' value is passed True in function call
            if scale:
                jac[jQOI, iPOI] *= xBase[iPOI] * np.sign(yBase[jQOI]) / (sys.float_info.epsilon + yBase[jQOI])
                                                                            # Scale jacobian for relative sensitivity
        del xPert, yPert, iPOI, jQOI                                        # Clear intermediate variables
    return jac                                                              # Return Jacobian

##--------------------------------------------------------------------------------------------------
def GetActiveSubspace(model,lsaOptions):
    eliminate=True
    inactiveIndex=np.zeros(model.nPOIs)
    #Calculate Jacobian
    jac=GetJacobian(model.evalFcn, model.basePOIs, lsaOptions, scale=False, yBase=model.baseQOIs)
    while eliminate:
        #Caclulate Fisher
        fisherMat=np.dot(np.transpose(jac), jac)
        #Perform Eigendecomp
        eigenValues, eigenVectors =np.linalg.eig(fisherMat)
        #Eliminate dimension/ terminate
        if np.min(eigenValues) < lsaOptions.subspaceRelTol * np.max(eigenValues):
            #Get inactive parameter
            inactiveParamReducedIndex=np.argmax(np.absolute(eigenVectors[:, np.argmin(np.absolute(eigenValues))]))
            inactiveParam=inactiveParamReducedIndex+np.sum(inactiveIndex[0:(inactiveParamReducedIndex+1)]).astype(int)
                #This indexing may seem odd but its because we're keeping the full model parameter numbering while trying
                # to index within the reduced model so we have to add to the index the previously removed params
            #Record inactive param in inactive space
            inactiveIndex[inactiveParam]=1
            #Remove inactive elements of jacobian
            jac=np.delete(jac,inactiveParamReducedIndex,1)
        else:
            #Terminate Active Subspace if singular values within tolerance
            eliminate=False
    #Define active and inactive spaces
    activeSpace=model.POInames[inactiveIndex == False]
    inactiveSpace=model.POInames[inactiveIndex == True]
    #Create Reduced model
    reducedModel=model.copy()
    # reducedModel.basePOIs=reducedModel.basePOIs[inactiveIndex == False]
    # reducedModel.POInames=reducedModel.POInames[inactiveIndex == False]
    # reducedModel.evalFcn = lambda reducedPOIs: model.evalFcn(
    #     np.array([x for x, y in zip(reducedPOIs,model.basePOIs) if inactiveIndex== True]))
    # #reducedModel.evalFcn=lambda reducedPOIs: model.evalFcn(np.where(inactiveIndex==False, reducedPOIs, model.basePOIs))
    # reducedModel.baseQOIs=reducedModel.evalFcn(reducedModel.basePOIs)
    return reducedModel, activeSpace, inactiveSpace

def ModelReduction(reducedModel,inactiveParam,model):
    #Record Index of reduced param
    inactiveIndex=np.where(reducedModel.POInames==inactiveParam)[0]
    #confirm exactly parameter matches
    if len(inactiveIndex)!=1:
        raise Exception("More than one or no POIs were found matching that name.")
    #Remove relevant data elements
    reducedModel.basePOIs=np.delete(reducedModel.basePOIs, inactiveIndex)
    reducedModel.POInames=np.delete(reducedModel.POInames, inactiveIndex)
    reducedModel.evalFcn=lambda reducedPOIs: model.evalFcn(np.where(inactiveIndex==True,reducedPOIs,model.basePOIs))
    print('made evalFcn')
    print(reducedModel.evalFcn(reducedModel.basePOIs))
    return reducedModel

def GetReducedPOIs(reducedPOIs,droppedIndices,model):
    fullPOIs=model.basePOIs
    reducedCounter=0
    print(droppedIndices)
    for i in np.arange(0,model.nPOIs):
        print(i)
        if droppedIndices==i:
            fullPOIs[i]=reducedPOIs[reducedCounter]
            reducedCounter=reducedCounter+1
    print(fullPOIs)
    return fullPOIs


##--------------------------------------GetSobol------------------------------------------------------
# GSA Component Functions

def GetSamples(model,gsaOptions):
    nSampSobol = gsaOptions.nSampSobol
    # Make 2 POI sample matrices with nSampSobol samples each
    if model.dist.lower()=='uniform' or model.dist.lower()=='saltellinormal':
        (sampA, sampB)=model.sampDist(nSampSobol);                                     #Get both A and B samples so no repeated values
    else:
        sampA = model.sampDist(nSampSobol)
        sampB = model.sampDist(nSampSobol)
    # Calculate matrices of QOI values for each POI sample matrix
    fA = model.evalFcn(sampA).reshape([nSampSobol, model.nQOIs])  # nSampSobol x nQOI out matrix from A
    fB = model.evalFcn(sampB).reshape([nSampSobol, model.nQOIs])  # nSampSobol x nQOI out matrix from B
    # Stack the output matrices into a single matrix
    fD = np.concatenate((fA.copy(), fB.copy()), axis=0)

    # Initialize combined QOI sample matrices
    if model.nQOIs == 1:
        fAB = np.empty([nSampSobol, model.nPOIs])
    else:
        fAB = np.empty([nSampSobol, model.nPOIs, model.nQOIs])
    for iParams in range(0, model.nPOIs):
        # Define sampC to be A with the ith parameter in B
        sampAB = sampA.copy()
        sampAB[:, iParams] = sampB[:, iParams].copy()
        if model.nQOIs == 1:
            fAB[:, iParams] = model.evalFcn(sampAB)
        else:
            fAB[:, iParams, :] = model.evalFcn(sampAB)  # nSampSobol x nPOI x nQOI tensor
        del sampAB
    return fA, fB, fAB, fD, np.concatenate((sampA.copy(), sampB.copy()), axis=0)

def CalculateSobol(fA, fB, fAB, fD):
    #Calculates calculates sobol indices using satelli approximation method
    #Inputs: model object (with evalFcn, sampDist, and nParams)
    #        sobolOptions object
    #Determing number of samples, QOIs, and POIs based on inputs
    nSampSobol=fAB.shape[0]
    if fAB.ndim==1:
        nQOIs=1
        nPOIs=1
    elif fAB.ndim==2:
        nQOIs=1
        nPOIs=fAB.shape[1]
    elif fAB.ndim==3:
        nPOIs=fAB.shape[1]
        nQOIs=fAB.shape[2]
    else:
        raise(Exception('fAB has greater than 3 dimensions, make sure fAB is the squeezed form of nSampSobol x nPOI x nQOI'))
    #QOI variance
    fDvar=np.var(fD, axis=0)

    sobolBase=np.empty((nQOIs, nPOIs))
    sobolTot=np.empty((nQOIs, nPOIs))
    if nQOIs==1:
        #Calculate 1st order parameter effects
        sobolBase=np.mean(fB*(fAB-fA), axis=0)/(fDvar)

        #Caclulate 2nd order parameter effects
        sobolTot=np.mean((fA-fAB)**2, axis=0)/(2*fDvar)

    else:
        for iQOI in range(0,nQOIs):
            #Calculate 1st order parameter effects
            sobolBase[iQOI,:]=np.mean(fB[:,[iQOI]]*(fAB[:,:,iQOI]-fA[:,[iQOI]]),axis=0)/fDvar[iQOI]
            #Caclulate 2nd order parameter effects
            sobolTot[iQOI,:]= np.mean((fA[:,[iQOI]]-fAB[:,:,iQOI])**2,axis=0)/(2*fDvar[iQOI])


    return sobolBase, sobolTot

##-------------------------------------GetMorris-------------------------------------------------------
def CaclulateMorris(model,options):
    #Define delta
    delta=(options.gsa.lMorris+1)/(2*options.gsa.lMorris)
    #Get Parameter Samples- use parameter distribution
    paramsSamp=model.sampDist(options.gsa.nSampMorris)[0]
    #Calulate derivative indices
    d= np.empty((options.gsa.nSampMorris, model.nPOIs, model.nQOIs)) #nQOIs x nPOIs x nSamples
    #Define constant sampling matrices
    J=np.ones((model.nPOIs+1,model.nPOIs))
    B = (np.tril(np.ones(J.shape), -1))
    for iSamp in range(0,options.gsa.nSampMorris):
        #Define Random Sampling matrices
        D=np.diag(np.random.choice(np.array([1,-1]), size=(model.nPOIs,)))
        P=np.identity(model.nPOIs)
        #np.random.shuffle(P)
        jTheta=paramsSamp[iSamp,]*J
        #CalculateMorris Sample matrix
        Bj=np.matmul(jTheta+delta/2*(np.matmul((2*B-J),D)+J),P)
        fBj=model.evalFcn(Bj)
        for k in np.arange(0,model.nPOIs):
            i=np.nonzero(Bj[k+1,:]-Bj[k,:])[0][0]
            print(np.nonzero(Bj[k+1,:]-Bj[k,:]))
            if Bj[k+1,i]-Bj[k,i]>0:
                if model.nQOIs==1:
                    d[iSamp,i]=(fBj[k+1]-fBj[k])/delta
                else:
                    d[iSamp,i,:]=(fBj[k+1]-fBj[k])/delta
            elif Bj[k+1,i]-Bj[k,i]<0:
                if model.nQOIs==1:
                    d[iSamp,i]=(fBj[k]-fBj[k+1])/delta
                else:
                    d[iSamp,i,:]=(fBj[k,:]-fBj[k+1,:])/delta
            else:
                raise(Exception('0 difference identified in Morris'))
    #Compute Indices- all outputs are nQOIs x nPOIs
    muStar=np.mean(np.abs(d),axis=0)
    sigma2=np.var(d, axis=0)

    return muStar, sigma2


##--------------------------------------GetSampDist----------------------------------------------------
def GetSampDist(model, gsaOptions):
    # Determine Sample Function- Currently only 1 distribution type can be defined for all parameters
    if model.dist.lower() == 'normal':  # Normal Distribution
        sampDist = lambda nSampSobol: np.random.randn(nSampSobol,model.nPOIs)*np.sqrt(model.distParms[[1], :]) + model.distParms[[0], :]
    elif model.dist.lower() == 'saltellinormal':
        sampDist = lambda nSampSobol: SaltelliNormal(nSampSobol, model.distParms)
    elif model.dist.lower() == 'uniform':  # uniform distribution
        # doubleParms=np.concatenate(model.distParms, model.distParms, axis=1)
        sampDist = lambda nSampSobol: SaltelliSample(nSampSobol,model.distParms)
    elif model.dist.lower() == 'exponential': # exponential distribution
        sampDist = lambda nSampSobol: np.random.exponential(model.distParms,size=(nSampSobol,model.nPOIs))
    elif model.dist.lower() == 'beta': # beta distribution
        sampDist = lambda nSampSobol:np.random.beta(model.distParms[[0],:], model.distParms[[1],:],\
                                               size=(nSampSobol,model.nPOIs))
    elif model.dist.lower() == 'InverseCDF': #Arbitrary distribution given by inverse cdf
        sampDist = lambda nSampSobol: gsaOptions.fInverseCDF(np.random.rand(nSampSobol,model.nPOIs))
    else:
        raise Exception("Invalid value for options.gsa.dist. Supported distributions are normal, uniform, exponential, beta, \
        and InverseCDF")  # Raise Exception if invalide distribution is entered
    model.sampDist=sampDist
    return model


#
#
def PlotGSA(model, sampleMat, evalMat, options):
    #Reduce Sample number
    #plotPoints=range(0,int(sampleMat.shape[0]), int(sampleMat.shape[0]/plotOptions.nPoints))
    #Make the number of sample points to survey
    plotPoints=np.linspace(start=0, stop=sampleMat.shape[0]-1, num=options.plot.nPoints, dtype=int)
    #Plot POI-POI correlation and distributions
    figure, axes=plt.subplots(nrows=model.nPOIs, ncols= model.nPOIs, squeeze=False)
    for iPOI in range(0,model.nPOIs):
        for jPOI in range(0,iPOI+1):
            if iPOI==jPOI:
                n, bins, patches = axes[iPOI, jPOI].hist(sampleMat[:,iPOI], bins=41)
            else:
                axes[iPOI, jPOI].plot(sampleMat[plotPoints,iPOI], sampleMat[plotPoints,jPOI],'b*')
            if jPOI==0:
                axes[iPOI,jPOI].set_ylabel(model.POInames[iPOI])
            if iPOI==model.nPOIs-1:
                axes[iPOI,jPOI].set_xlabel(model.POInames[jPOI])
            if model.nPOIs==1:
                axes[iPOI,jPOI].set_ylabel('Instances')
    figure.tight_layout()
    if options.path:
        plt.savefig(options.path+"POIcorrelation.png")

    #Plot QOI-QOI correlationa and distributions
    figure, axes=plt.subplots(nrows=model.nQOIs, ncols= model.nQOIs, squeeze=False)
    for iQOI in range(0,model.nQOIs):
        for jQOI in range(0,iQOI+1):
            if iQOI==jQOI:
                axes[iQOI, jQOI].hist([evalMat[:,iQOI]], bins=41)
            else:
                axes[iQOI, jQOI].plot(evalMat[plotPoints,iQOI], evalMat[plotPoints,jQOI],'b*')
            if jQOI==0:
                axes[iQOI,jQOI].set_ylabel(model.QOInames[iQOI])
            if iQOI==model.nQOIs-1:
                axes[iQOI,jQOI].set_xlabel(model.QOInames[jQOI])
            if model.nQOIs==1:
                axes[iQOI,jQOI].set_ylabel('Instances')
    figure.tight_layout()
    if options.path:
        plt.savefig(options.path+"QOIcorrelation.png")

    #Plot POI-QOI correlation
    figure, axes=plt.subplots(nrows=model.nQOIs, ncols= model.nPOIs, squeeze=False)
    for iQOI in range(0,model.nQOIs):
        for jPOI in range(0, model.nPOIs):
            axes[iQOI, jPOI].plot(sampleMat[plotPoints,jPOI], evalMat[plotPoints,iQOI],'b*')
            if jPOI==0:
                axes[iQOI,jPOI].set_ylabel(model.QOInames[iQOI])
            if iQOI==model.nQOIs-1:
                axes[iQOI,jPOI].set_xlabel(model.POInames[jPOI])
    if options.path:
        plt.savefig(options.path+"POI_QOIcorrelation.png")
    #Display all figures
    if options.display:
        plt.show()

def SaltelliSample(nSampSobol,distParams):
    nPOIs=distParams.shape[1]
    baseSample=sobol.sample(dimension=nPOIs*2, n_points=nSampSobol, skip=1099)
    baseA=baseSample[:,:nPOIs]
    baseB=baseSample[:,nPOIs:2*nPOIs]
    sampA=distParams[[0],:]+(distParams[[1],:]-distParams[[0],:])*baseA
    sampB=distParams[[0],:]+(distParams[[1],:]-distParams[[0],:])*baseB
    return (sampA, sampB)

def SaltelliNormal(nSampSobol, distParms):
    nPOIs=distParms.shape[1]
    baseSample=sobol.sample(dimension=nPOIs*2, n_points=nSampSobol, skip=1099)
    baseA=baseSample[:,:nPOIs]
    baseB=baseSample[:,nPOIs:2*nPOIs]
    transformA=sct.norm.ppf(baseA)
    transformB=sct.norm.ppf(baseB)
    sampA=transformA*np.sqrt(distParms[[1], :]) + distParms[[0], :]
    sampB=transformB*np.sqrt(distParms[[1], :]) + distParms[[0], :]
    return (sampA, sampB)


def TestAccuracy(model,options,nSampSobol):
    baseSobol=np.empty((nSampSobol.size, model.nPOIs))
    totalSobol=np.empty((nSampSobol.size, model.nPOIs))
    options.plot.run=False
    options.lsa.run=False
    options.print=False
    for iSamp in range(0,nSampSobol.size):
        options.gsa.nSampSobol=nSampSobol[iSamp]
        results=RunUQ(model,options)
        baseSobol[iSamp,:]=results.gsa.sobolBase
        totalSobol[iSamp,:]=results.gsa.sobolTot
    figure, axes=plt.subplots(nrows=2, ncols= model.nPOIs, squeeze=False)
    for iPOI in np.arange(0,model.nPOIs):
        axes[0, iPOI].plot(nSampSobol, baseSobol[:,iPOI], 'bs')
        axes[1, iPOI].plot(nSampSobol, totalSobol[:,iPOI], 'bs')
        axes[0,iPOI].set_title(model.POInames[iPOI])
    axes[0,0].set_ylabel('First Order Sobol')
    axes[1,0].set_ylabel('Total Sobol')
    axes[1,0].set_xlabel('Number of Samples')
    axes[1,1].set_xlabel('Number of Samples')
    figure.tight_layout()
    if options.path:
        plt.savefig(options.path+"SobolConvergence.png")
    plt.show()
    return (baseSobol,totalSobol)
