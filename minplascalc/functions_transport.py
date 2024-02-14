import numpy
from scipy import constants

pi = constants.pi
ke = 1/(4*pi*constants.epsilon_0)
kb = constants.Boltzmann
kav = constants.Avogadro
qe = constants.elementary_charge
k2e = 1/constants.physical_constants['electron volt-kelvin relationship'][0]
kR = constants.gas_constant
gamma = numpy.euler_gamma

c0_nn_11 = [7.884756e-1, -2.952759e-1, 5.020892e-1, -9.042460e-1, -3.373058, 
            4.161981, 2.462523]
c1_nn_11 = [-2.438494e-2, -1.744149e-3, 4.316985e-2, -4.017103e-2, 2.458538e-1, 
            2.202737e-1, 3.231308e-1]
c2_nn_11 = [0, 0, 0, 0, -4.850047e-3, -1.718010e-2, -2.281072e-2]
c0_nn_12 = [7.123565e-1, -2.910530e-1, 4.187065e-2, -9.287685e-1, -3.598542, 
            3.934824, 2.578084]
c1_nn_12 = [-2.688875e-2, -2.065175e-3, 4.060236e-2, -2.342270e-2, 2.545120e-1, 
            2.699944e-1, 3.449024e-1]
c2_nn_12 = [0, 0, 0, 0, -4.685966e-3, -2.009886e-2, -2.292710e-2]
c0_nn_13 = [6.606022e-1, -2.870900e-1, -2.519690e-1, -9.173046e-1, -3.776812, 
            3.768103, 2.695440]
c1_nn_13 = [-2.831448e-2, -2.232827e-3, 3.778211e-2, -1.864476e-2, 2.552528e-1, 
            3.155025e-1, 3.597998e-1]
c2_nn_13 = [0, 0, 0, 0, -4.237220e-3, -2.218849e-2, -2.267102e-2]
c0_nn_14 = [6.268016e-1, -2.830834e-1, -4.559927e-1, -9.334638e-1, -3.947019, 
            3.629926, 2.824905]
c1_nn_14 = [-2.945078e-2, -2.361273e-3, 3.705640e-2, -1.797329e-2, 2.446843e-1, 
            3.761272e-1, 3.781709e-1]
c2_nn_14 = [0, 0, 0, 0, -3.176374e-3, -2.451016e-2, -2.251978e-2]
c0_nn_15 = [5.956859e-1, -2.804989e-1, -5.965551e-1, -8.946001e-1, -4.076798, 
            3.458362, 2.982260]
c1_nn_15 = [-2.915893e-2, -2.298968e-3, 3.724395e-2, -2.550731e-2, 1.983892e-1, 
            4.770695e-1, 4.014572e-1]
c2_nn_15 = [0, 0, 0, 0, -5.014065e-4, -2.678054e-2, -2.142580e-2]
c0_nn_22 = [7.898524e-1, -2.998325e-1, 7.077103e-1, -8.946857e-1, -2.958969, 
            4.348412, 2.205440]
c1_nn_22 = [-2.114115e-2, -1.243977e-3, 3.583907e-2, -2.473947e-2, 2.303358e-1, 
            1.920321e-1, 2.567027e-1]
c2_nn_22 = [0, 0, 0, 0, -5.226562e-3, -1.496557e-2, -1.861359e-2]
c0_nn_23 = [7.269006e-1, -2.972304e-1, 3.904230e-1, -9.442201e-1, -3.137828, 
            4.190370, 2.319751]
c1_nn_23 = [-2.233866e-2, -1.392888e-3, 3.231655e-2, -1.494805e-2, 2.347767e-1, 
            2.346004e-1, 2.700236e-1]
c2_nn_23 = [0, 0, 0, 0, -4.963979e-3, -1.718963e-2, -1.854217e-2]
c0_nn_24 = [6.829159e-1, -2.943232e-1, 1.414623e-1, -9.720228e-1, -3.284219, 
            4.011692, 2.401249]
c1_nn_24 = [-2.332763e-2, -1.514322e-3, 3.075351e-2, -1.038869e-2, 2.243767e-1, 
            3.005083e-1, 2.943600e-1]
c2_nn_24 = [0, 0, 0, 0, -3.913041e-3, -2.012373e-2, -1.884503e-2]
c0_nn_33 = [7.468781e-1, -2.947438e-1, 2.234096e-1, -9.974591e-1, -3.381787, 
            4.094540, 2.476087]
c1_nn_33 = [-2.518134e-2, -1.811571e-3, 3.681114e-2, -2.670805e-2, 2.372932e-1, 
            2.756466e-1, 3.300898e-1]
c2_nn_33 = [0, 0, 0, 0, -4.239629e-3, -2.009227e-2, -2.223317e-2]
c0_nn_44 = [7.365470e-1, -2.968650e-1, 3.747555e-1, -9.944036e-1, -3.136655, 
            4.145871, 2.315532]
c1_nn_44 = [-2.242357e-2, -1.396696e-3, 2.847063e-2, -1.378926e-2, 2.176409e-1, 
            2.855836e-1, 2.842981e-1]
c2_nn_44 = [0, 0, 0, 0, -3.899247e-3, -1.939452e-2, -1.874462e-2]
c_nn_11 = numpy.array([c0_nn_11, c1_nn_11, c2_nn_11]).transpose()
c_nn_12 = numpy.array([c0_nn_12, c1_nn_12, c2_nn_12]).transpose()
c_nn_13 = numpy.array([c0_nn_13, c1_nn_13, c2_nn_13]).transpose()
c_nn_14 = numpy.array([c0_nn_14, c1_nn_14, c2_nn_14]).transpose()
c_nn_15 = numpy.array([c0_nn_15, c1_nn_15, c2_nn_15]).transpose()
c_nn_22 = numpy.array([c0_nn_22, c1_nn_22, c2_nn_22]).transpose()
c_nn_23 = numpy.array([c0_nn_23, c1_nn_23, c2_nn_23]).transpose()
c_nn_24 = numpy.array([c0_nn_24, c1_nn_24, c2_nn_24]).transpose()
c_nn_33 = numpy.array([c0_nn_33, c1_nn_33, c2_nn_33]).transpose()
c_nn_44 = numpy.array([c0_nn_44, c1_nn_44, c2_nn_44]).transpose()
fillnan = numpy.full(c_nn_11.shape, numpy.nan)
c_nn = numpy.array([[c_nn_11, c_nn_12, c_nn_13, c_nn_14, c_nn_15], 
                    [fillnan, c_nn_22, c_nn_23, c_nn_24, fillnan],
                    [fillnan, fillnan, c_nn_33, fillnan, fillnan],
                    [fillnan, fillnan, fillnan, c_nn_44, fillnan]])

c0_in_11 = [9.851755e-1, -4.737800e-1, 7.080799e-1, -1.239439, -4.638467, 
            3.841835, 2.317342]
c1_in_11 = [-2.870704e-2, -1.370344e-3, 4.575312e-3, -3.683605e-2, 4.418904e-1, 
            3.277658e-1, 3.912768e-1]
c2_in_11 = [0, 0, 0, 0, -1.220292e-2, -2.660275e-2, -3.136223e-2]
c0_in_12 = [8.361751e-1, -4.707355e-1, 1.771157e-1, -1.094937, -4.976384, 
            3.645873, 2.428864]
c1_in_12 = [-3.201292e-2, -1.783284e-3, 1.172773e-2, -3.115598e-2, 4.708074e-1, 
            3.699452e-1, 4.267351e-1]
c2_in_12 = [0, 0, 0, 0, -1.283818e-2, -2.988684e-2, -3.278874e-2]
c0_in_13 = [7.440562e-1, -4.656306e-1, -1.465752e-1, -1.080410, -5.233907, 
            3.489814, 2.529678]
c1_in_13 = [-3.453851e-2, -2.097901e-3, 1.446209e-2, -2.712029e-2, 4.846691e-1, 
            4.140270e-1, 4.515088e-1]
c2_in_13 = [0, 0, 0, 0, -1.280346e-2, -3.250138e-2, -3.339293e-2]
c0_in_14 = [6.684360e-1, -4.622014e-1, -3.464990e-1, -1.054374, -5.465789, 
            3.374614, 2.648622]
c1_in_14 = [-3.515695e-2, -2.135808e-3, 1.336362e-2, -3.149321e-2, 4.888443e-1, 
            4.602468e-1, 4.677409e-1]
c2_in_14 = [0, 0, 0, 0, -1.228090e-2, -3.463073e-2, -3.339297e-2]
c0_in_15 = [6.299083e-1, -4.560921e-1, -5.228598e-1, -1.124725, -5.687354, 
            3.267709, 2.784725]
c1_in_15 = [-3.720000e-2, -2.395779e-3, 1.594610e-2, -2.862354e-2, 4.714789e-1, 
            5.281419e-1, 4.840700e-1]
c2_in_15 = [0, 0, 0, 0, -1.056602e-2, -3.678869e-2, -3.265127e-2]
c0_in_22 = [9.124518e-1, -4.697184e-1, 1.031053, -1.090782, -4.127243, 4.059078,
            2.086906]
c1_in_22 = [-2.398461e-2, -7.809681e-4, 4.069668e-3, -2.413508e-2, 4.302667e-1, 
            2.597379e-1, 2.920310e-1]
c2_in_22 = [0, 0, 0, 0, -1.352874e-2, -2.169951e-2, -2.560437e-2]
c0_in_23 = [8.073459e-1, -4.663682e-1, 6.256342e-1, -1.063437, -4.365989, 
            3.854346, 2.146207]
c1_in_23 = [-2.581232e-2, -1.030271e-3, 4.086881e-3, -1.235489e-2, 4.391454e-1, 
            3.219224e-1, 3.325620e-1]
c2_in_23 = [0, 0, 0, 0, -1.314615e-2, -2.587493e-2, -2.686959e-2]
c0_in_24 = [7.324117e-1, -4.625614e-1, 3.315871e-1, -1.055706, -4.571022, 
            3.686006, 2.217893]
c1_in_24 = [-2.727580e-2, -1.224292e-3, 7.216776e-3, -8.585500e-3, 4.373660e-1, 
            3.854493e-1, 3.641196e-1]
c2_in_24 = [0, 0, 0, 0, -1.221457e-2, -2.937568e-2, -2.763824e-2]
c0_in_33 = [8.402943e-1, -4.727437e-1, 4.724228e-1, -1.213660, -4.655574, 
            3.817178, 2.313186]
c1_in_33 = [-2.851694e-2, -1.328784e-3, 7.706027e-3, -3.456656e-2, 4.467685e-1, 
            3.503180e-1, 3.889828e-1]
c2_in_33 = [0, 0, 0, 0, -1.237864e-2, -2.806506e-2, -3.120619e-2]
c0_in_44 = [8.088842e-1, -4.659483e-1, 6.092981e-1, -1.113323, -4.349145, 
            3.828467, 2.138075]
c1_in_44 = [-2.592379e-2, -1.041599e-3, 1.428402e-3, -1.031574e-2, 4.236246e-1, 
            3.573461e-1, 3.388072e-1]
c2_in_44 = [0, 0, 0, 0, -1.210668e-2, -2.759622e-2, -2.669344e-2]
c_in_11 = numpy.array([c0_in_11, c1_in_11, c2_in_11]).transpose()
c_in_12 = numpy.array([c0_in_12, c1_in_12, c2_in_12]).transpose()
c_in_13 = numpy.array([c0_in_13, c1_in_13, c2_in_13]).transpose()
c_in_14 = numpy.array([c0_in_14, c1_in_14, c2_in_14]).transpose()
c_in_15 = numpy.array([c0_in_15, c1_in_15, c2_in_15]).transpose()
c_in_22 = numpy.array([c0_in_22, c1_in_22, c2_in_22]).transpose()
c_in_23 = numpy.array([c0_in_23, c1_in_23, c2_in_23]).transpose()
c_in_24 = numpy.array([c0_in_24, c1_in_24, c2_in_24]).transpose()
c_in_33 = numpy.array([c0_in_33, c1_in_33, c2_in_33]).transpose()
c_in_44 = numpy.array([c0_in_44, c1_in_44, c2_in_44]).transpose()
fillnan = numpy.full(c_in_11.shape, numpy.nan)
c_in = numpy.array([[c_in_11, c_in_12, c_in_13, c_in_14, c_in_15], 
                    [fillnan, c_in_22, c_in_23, c_in_24, fillnan],
                    [fillnan, fillnan, c_in_33, fillnan, fillnan],
                    [fillnan, fillnan, fillnan, c_in_44, fillnan]])

def n_effective_electrons(nint, nout):
    return nout * (1 + (1 - nout/nint) * (nint/(nout+nint))**2)

def pot_parameters_neut_neut(spi, spj):
    ai, aj = spi.polarisability*1e30, spj.polarisability*1e30
    londoncoefft = (15.7*ai*aj / (numpy.sqrt(ai/spi.effectiveelectrons) 
                                  + numpy.sqrt(aj/spj.effectiveelectrons)))
    re = (1.767*(ai**(1/3) + aj**(1/3)) / (ai*aj)**0.095)
    e0 = 0.72*londoncoefft / re**6
    return re, e0

def pot_parameters_ion_neut(spi, spn):
    ai, an = spi.polarisability*1e30, spn.polarisability*1e30
    rho = ai / (spi.chargenumber**2 * numpy.sqrt(an) 
                * (1 + (2*ai/an)**(2/3)))
    re = (1.767*(ai**(1/3) + an**(1/3)) / (ai*an*(1+1/rho))**0.095)
    e0 = 5.2*spi.chargenumber**2*an*(1+rho) / re**4
    return re, e0    

def beta(spi, spj):
    ai, aj = spi.polarisability*1e30, spj.polarisability*1e30
    return 6 + 5 / (spi.multiplicity*ai**(1/3) + spj.multiplicity*aj**(1/3))

def x0_neut_neut(betaval):
    return 0.8002 * betaval**0.049256

def x0_ion_neut(betaval):
    return 0.7564 * betaval**0.064605

def cl_charged(spi, spj, ni, nj, T):
    T_eV = T * k2e
    if spi.name == 'e' and spj.name == 'e':
        ne_cgs = ni * 1e-6
        return (23.5 - numpy.log(ne_cgs**(1/2) * T_eV **(-5/4)) 
                - (1e-5 + (numpy.log(T_eV)-2)**2 / 16)**(1/2))
    elif spi.name == 'e':
        ne_cgs = ni * 1e-6
        return 23 - numpy.log(ne_cgs**(1/2) * spj.chargenumber * T_eV**(-3/2))
    elif spj.name == 'e':
        ne_cgs = nj * 1e-6
        return 23 - numpy.log(ne_cgs**(1/2) * spi.chargenumber * T_eV**(-3/2))
    else:
        ni_cgs, nj_cgs = ni * 1e-6, nj * 1e-6
        return 23 - numpy.log(spi.chargenumber*spj.chargenumber/T_eV 
                              * (ni_cgs*spi.chargenumber**2/T_eV 
                              + nj_cgs*spj.chargenumber**2/T_eV)**(1/2))

def psiconst(s):
    if s == 1:
        return 0
    else:
        return numpy.sum(1/numpy.arange(1,s))
    
def A(ie):
    ie_eV = ie / qe
    return numpy.sqrt(pi) * 9.81867945e-09 / ie_eV**0.729218856

def B(ie):
    ie_eV = ie / qe
    return numpy.sqrt(pi) * 4.78257679e-10 / ie_eV**0.657012657

def sum1(s):
    return numpy.sum(1/numpy.array(range(1,s+2))) - gamma

def sum2(s):
    return numpy.sum(1/numpy.array(range(1,s+2))**2)

def delta(i, j):
    if i==j:
        return 1
    else:
        return 0

### Collision cross section calculations #######################################

def Qe(spi):
    ''' Electron-neutral collision cross section.
    '''
    return spi.electroncrosssection

def Qnn(spi, spj, l, s, T):
    ''' Neutral-neutral elastic collision cross section.
    '''
    if ((l == 1 and s >= 6) or (l == 2 and s >= 5) or (l == 3 and s >= 4) 
        or (l == 4 and s >= 5)):
        negT, posT = T - 0.5, T + 0.5
        return (Qnn(spi,spj,l,s-1,T) + T/(s+1) * (Qnn(spi,spj,l,s-1,posT)
                - Qnn(spi,spj,l,s-1,negT)))
    re, e0 = pot_parameters_neut_neut(spi, spj)
    bv = beta(spi, spj)
    x0 = x0_neut_neut(bv)
    a = c_nn[l-1, s-1].dot([1, bv, bv**2])
    sigma = re*x0
    x = numpy.log(k2e*T/e0)
    lnS1 = ((a[0]+a[1]*x) * numpy.exp((x-a[2])/a[3]) 
            / (numpy.exp((x-a[2])/a[3]) + numpy.exp((a[2]-x)/a[3])))
    lnS2 = (a[4] * numpy.exp((x-a[5])/a[6]) 
            / (numpy.exp((x-a[5])/a[6]) + numpy.exp((a[5]-x)/a[6])))
    return numpy.exp(lnS1+lnS2) * pi * sigma**2 * 1e-20

def Qin(spi, spj, l, s, T):
    ''' Ion-neutral elastic collision cross section.
    '''
    if ((l == 1 and s >= 6) or (l == 2 and s >= 5) or (l == 3 and s >= 4) 
        or (l == 4 and s >= 5)):
        negT, posT = T - 0.5, T + 0.5
        return (Qin(spi,spj,l,s-1,T) + T/(s+1) * (Qin(spi,spj,l,s-1,posT) 
                - Qin(spi,spj,l,s-1,negT)))
    re, e0 = pot_parameters_ion_neut(spi, spj)
    bv = beta(spi, spj)
    x0 = x0_ion_neut(bv)
    a = c_in[l-1, s-1].dot([1, bv, bv**2])
    sigma = re*x0
    x = numpy.log(k2e*T/e0)
    lnS1 = ((a[0]+a[1]*x) * numpy.exp((x-a[2])/a[3]) 
            / (numpy.exp((x-a[2])/a[3]) + numpy.exp((a[2]-x)/a[3])))
    lnS2 = (a[4] * numpy.exp((x-a[5])/a[6]) 
            / (numpy.exp((x-a[5])/a[6]) + numpy.exp((a[5]-x)/a[6])))
    return numpy.exp(lnS1+lnS2) * pi * sigma**2 * 1e-20

# Ion-neutral resonant charge transfer cross section
def Qtr(spi, spj, s, T):
    ''' Ion-neutral resonant charge transfer cross section.
    '''
    if spi.chargenumber == 0:     
        a, b = A(spi.ionisationenergy), B(spi.ionisationenergy)
        mm = spi.molarmass
    else:
        a, b = A(spj.ionisationenergy), B(spj.ionisationenergy)
        mm = spj.molarmass
    lnterm = numpy.log(4*kR*T/mm)
    s1, s2 = sum1(s), sum2(s)
    cterm = (pi**2/6 - s2 + s1**2)
    return (a**2 - s1*a*b + (b/2)**2 * cterm + (b/2)**2 * lnterm**2 
            + (s1*b**2/2 - a*b)*lnterm)

def Qc(spi, ni, spj, nj, l, s, T):
    ''' Coulomb collision cross section. 
    '''
    preconst = [4, 12, 12, 16]
    addconst = [1/2, 1, 7/6, 4/3]
    term1 = preconst[l-1] * pi / (s * (s + 1))
    term2 = (ke * spi.chargenumber * spj.chargenumber * qe**2 
             / (2 * kb * T)) ** 2
    term3 = (cl_charged(spi, spj, ni, nj, T) + numpy.log(2) - addconst[l-1] 
             - 2*gamma + psiconst(s))
    return term1 * term2 * term3

### Unified cross section calculations #########################################

def Qij(spi, ni, spj, nj, l, s, T):
    if spi.chargenumber != 0 and spj.chargenumber != 0:
        return Qc(spi, ni, spj, nj, l, s, T)
    elif spj.name == 'e':
        return Qe(spi)
    elif spi.name == 'e':
        return Qe(spj)
    elif spi.chargenumber == 0 and spj.chargenumber == 0:
        return Qnn(spi, spj, l, s, T)
    elif abs(spi.chargenumber-spj.chargenumber)==1 and l % 2 == 1:
        return Qtr(spi, spj, s, T)
    elif spi.chargenumber == 0:
        return Qin(spj, spi, l, s, T)
    elif spj.chargenumber == 0:
        return Qin(spi, spj, l, s, T)
    else:
        raise ValueError('Unknown collision type')
        
def Qij_mix(mix, l, s):
    Qvals = numpy.zeros((len(mix.species), len(mix.species)))
    numberdensities = mix.calculate_composition()
    for i, (ndi, spi) in enumerate(zip(numberdensities, mix.species)):
        for j, (ndj, spj) in enumerate(zip(numberdensities, mix.species)):
            Qvals[i,j] = Qij(spi, ndi, spj, ndj, l, s, mix.T)
    return Qvals


### q-matrix calculations ######################################################

def q(mix):
    nsp = len(mix.species)
    nv = mix.calculate_composition()
    mv = numpy.array([sp.molarmass / kav for sp in mix.species])

    Q11 = Qij_mix(mix, 1, 1)
    Q12 = Qij_mix(mix, 1, 2)
    Q13 = Qij_mix(mix, 1, 3)
    Q14 = Qij_mix(mix, 1, 4)
    Q15 = Qij_mix(mix, 1, 5)
    Q16 = Qij_mix(mix, 1, 6)
    Q17 = Qij_mix(mix, 1, 7)
    Q22 = Qij_mix(mix, 2, 2)
    Q23 = Qij_mix(mix, 2, 3)
    Q24 = Qij_mix(mix, 2, 4)
    Q25 = Qij_mix(mix, 2, 5)
    Q26 = Qij_mix(mix, 2, 6)
    Q33 = Qij_mix(mix, 3, 3)
    Q34 = Qij_mix(mix, 3, 4)
    Q35 = Qij_mix(mix, 3, 5)
    Q44 = Qij_mix(mix, 4, 4)   

    q00 = numpy.zeros((nsp, nsp))
    for i in range(nsp):
        for j in range(nsp):
            sumval = 0
            for l in range(nsp):
                term1 = nv[l]*mv[i]**(1/2) / (mv[i] + mv[l])**(1/2)
                term2 = (nv[i]*(mv[l]/mv[j])**(1/2)*(delta(i,j)-delta(j,l)) 
                         - nv[j]*(mv[l]*mv[j])**(1/2)/mv[i]*(1-delta(i,l)))
                sumval += term1*Q11[i,l]*term2
            q00[i,j] = 8*sumval

    q01 = numpy.zeros((nsp, nsp))
    for i in range(nsp):
        for j in range(nsp):
            sumval = 0
            for l in range(nsp):
                term1 = nv[l]*mv[l]**(3/2) / (mv[i] + mv[l])**(3/2)
                term2 = (delta(i,j)-delta(j,l)) * (5/2*Q11[i,l] - 3*Q12[i,l])
                sumval += term1*term2
            q01[i,j] = 8*nv[i]*(mv[i]/mv[j])**(3/2) * sumval

    q11 = numpy.zeros((nsp, nsp))
    for i in range(nsp):
        for j in range(nsp):
            sumval = 0
            for l in range(nsp):
                term1 = nv[l]*mv[l]**(1/2) / (mv[i] + mv[l])**(5/2)
                term2 = ((delta(i,j)-delta(j,l)) 
                         * (5/4*(6*mv[j]**2+5*mv[l]**2)*Q11[i,l] 
                            -15*mv[l]**2*Q12[i,l]+12*mv[l]**2*Q13[i,l]) 
                         + (delta(i,j)+delta(j,l))*4*mv[j]*mv[l]*Q22[i,l])
                sumval += term1*term2
            q11[i,j] = 8*nv[i]*(mv[i]/mv[j])**(3/2) * sumval

    q02 = numpy.zeros((nsp, nsp))
    for i in range(nsp):
        for j in range(nsp):
            sumval = 0
            for l in range(nsp):
                term1 = nv[l]*mv[l]**(5/2) / (mv[i] + mv[l])**(5/2)
                term2 = ((delta(i,j)-delta(j,l)) 
                         * (35/8*Q11[i,l]-21/2*Q12[i,l]+6*Q13[i,l]))
                sumval += term1*term2
            q02[i,j] = 8*nv[i]*(mv[i]/mv[j])**(5/2) * sumval

    q12 = numpy.zeros((nsp, nsp))
    for i in range(nsp):
        for j in range(nsp):
            sumval = 0
            for l in range(nsp):
                term1 = nv[l]*mv[l]**(3/2) / (mv[i] + mv[l])**(7/2)
                term2 = ((delta(i,j)-delta(j,l)) 
                         * (35/16*(12*mv[j]**2+5*mv[l]**2)*Q11[i,l] 
                            - 63/2*(mv[j]**2+5/4*mv[l]**2)*Q12[i,l]
                            + 57*mv[l]**2*Q13[i,l] - 30*mv[l]**2*Q14[i,l]) 
                         + (delta(i,j)+delta(j,l)) 
                         * (14*mv[j]*mv[l]*Q22[i,l] - 16*mv[j]*mv[l]*Q23[i,l]))
                sumval += term1*term2
            q12[i,j] = 8*nv[i]*(mv[i]/mv[j])**(5/2) * sumval

    q22 = numpy.zeros((nsp, nsp))
    for i in range(nsp):
        for j in range(nsp):
            sumval = 0
            for l in range(nsp):
                term1 = nv[l]*mv[l]**(1/2) / (mv[i] + mv[l])**(9/2)
                term2 = ((delta(i,j)-delta(j,l)) 
                         * (35/64*(40*mv[j]**4+168*(mv[j]*mv[l])**2
                                   +35*mv[l]**4)*Q11[i,l] 
                            - 21/8*mv[l]**2*(84*mv[j]**2+35*mv[l]**2)*Q12[i,l]
                            + 3/2*mv[l]**2*(108*mv[j]**2+133*mv[l]**2)*Q13[i,l]
                            - 210*mv[l]**4*Q14[i,l]+90*mv[l]**4*Q15[i,l]
                            + 24*(mv[j]*mv[l])**2*Q33[i,j]) 
                         + (delta(i,j)+delta(j,l)) 
                         * (7*mv[j]*mv[l]*(4*(mv[j]**2+7*mv[l]**2))*Q22[i,l]
                            - 112*mv[j]*mv[l]**3*Q23[i,l] 
                            + 80*mv[j]*mv[l]**3*Q24[i,l]))
                sumval += term1*term2
            q22[i,j] = 8*nv[i]*(mv[i]/mv[j])**(5/2) * sumval

    q03 = numpy.zeros((nsp, nsp))
    for i in range(nsp):
        for j in range(nsp):
            sumval = 0
            for l in range(nsp):
                term1 = nv[l]*mv[l]**(7/2) / (mv[i] + mv[l])**(7/2)
                term2 = ((delta(i,j)-delta(j,l)) 
                         * (105/16*Q11[i,l] - 189/8*Q12[i,l] + 27*Q13[i,l] 
                            - 10*Q14[i,l]))
                sumval += term1*term2
            q03[i,j] = 8*nv[i]*(mv[i]/mv[j])**(7/2) * sumval


    q13 = numpy.zeros((nsp, nsp))
    for i in range(nsp):
        for j in range(nsp):
            sumval = 0
            for l in range(nsp):
                term1 = nv[l]*mv[l]**(5/2) / (mv[i] + mv[l])**(9/2)
                term2 = ((delta(i,j)-delta(j,l)) 
                         * (105/32*(18*mv[j]**2+5*mv[l]**2)*Q11[i,l]
                            - 63/4*(9*mv[j]**2+5*mv[l]**2)*Q12[i,l] 
                            + 81*(mv[j]**2+2*mv[l]**2)*Q13[i,l]
                            - 160*mv[l]**2*Q14[i,l] + 60*mv[l]**2*Q15[i,l]) 
                         + (delta(i,j)+delta(j,l)) 
                         * mv[j]*mv[l]*(63/2*Q22[i,l] - 72*Q23[i,l] 
                                        + 40*Q24[i,l]))
                sumval += term1*term2
            q13[i,j] = 8*nv[i]*(mv[i]/mv[j])**(7/2) * sumval


    q23 = numpy.zeros((nsp, nsp))
    for i in range(nsp):
        for j in range(nsp):
            sumval = 0
            for l in range(nsp):
                term1 = nv[l]*mv[l]**(3/2) / (mv[i] + mv[l])**(11/2)
                term2 = ((delta(i,j)-delta(j,l)) 
                         * (105/128*(120*mv[j]**4+252*(mv[j]*mv[l])**2
                                     +35*mv[l]**4)*Q11[i,l] 
                            - 63/64*(120*mv[j]**4+756*(mv[j]*mv[l])**2
                                     +175*mv[l]**4)*Q12[i,l]
                            + 9/4*mv[l]**2*(450*mv[j]**2+217*mv[l]**2)*Q13[i,l]
                            + 5/2*mv[l]**2*(198*mv[j]**2+301*mv[l]**2)*Q14[i,l]
                            + 615*mv[l]**4*Q15[i,l] - 210*mv[l]**4*Q16[i,l]
                            + 108*(mv[j]*mv[l])**2*Q33[i,j] 
                            - 120*(mv[j]*mv[l])**2*Q34[i,j]) 
                         + (delta(i,j)+delta(j,l)) 
                         * (63/4*mv[j]*mv[l]*(8*(mv[j]**2+7*mv[l]**2))*Q22[i,l]
                            - 18*mv[j]*mv[l]*(8*mv[j]**2+21*mv[l]**2)*Q23[i,l] 
                            + 500*mv[j]*mv[l]**3*Q24[i,l] 
                            - 240*mv[j]*mv[l]**3*Q25[i,l]))
                sumval += term1*term2
            q23[i,j] = 8*nv[i]*(mv[i]/mv[j])**(7/2) * sumval
            
    q33 = numpy.zeros((nsp, nsp))
    for i in range(nsp):
        for j in range(nsp):
            sumval = 0
            for l in range(nsp):
                term1 = nv[l]*mv[l]**(1/2) / (mv[i] + mv[l])**(13/2)
                term2 = ((delta(i,j)-delta(j,l)) 
                         * (105/256*(112*mv[j]**6+1080*mv[j]**4*mv[l]**2
                                     +1134*mv[j]**2*mv[l]**4
                                     +105*mv[l]**6)*Q11[i,l]
                            - 567/64*mv[l]**2*(120*mv[j]**4+252*(mv[j]*mv[l])**2
                                               +35*mv[l]**4)*Q12[i,l]
                            + 27/16*mv[l]**2*(440*mv[j]**4+2700*(mv[j]*mv[l])**2
                                              +651*mv[l]**4)*Q13[i,l]
                            + 15/2*mv[l]**4*(594*mv[j]**2+301*mv[l]**2)*Q14[i,l]
                            + 135/2*mv[l]**4*(26*mv[j]**2+41*mv[l]**2)*Q15[i,l] 
                            - 1890*mv[l]**6*Q16[i,l] - 560*mv[l]**6*Q17[i,l] 
                            + 18*(mv[j]*mv[l])**2*(10*mv[j]**2
                                                   +27*mv[l]**2)*Q33[i,j]
                            - 1080*mv[j]**2*mv[l]**4*Q34[i,j] 
                            + 720*mv[j]**2*mv[l]**4*Q35[i,j]) 
                         + (delta(i,j)+delta(j,l)) 
                         * (189/16*mv[j]*mv[l]*(8*mv[j]**4+48*(mv[j]*mv[l])**2
                                                +21*mv[l]**4)*Q22[i,l]
                            - 162*mv[j]*mv[l]**3*(8*mv[j]**2
                                                  +7*mv[l]**2)*Q23[i,l]
                            + 10*mv[j]*mv[l]**3*(88*mv[j]**2
                                                 +225*mv[l]**2)*Q24[i,l]
                            - 2160*mv[j]*mv[l]**5*Q25[i,l] 
                            + 840*mv[j]*mv[l]**5*Q26[i,l] 
                            + 64*(mv[j]*mv[l])**3*Q44[i,l]))
                sumval += term1*term2
            q33[i,j] = 8*nv[i]*(mv[i]/mv[j])**(7/2) * sumval
            
    q10 = numpy.zeros((nsp, nsp))
    for i in range(nsp):
        for j in range(nsp):
            q10[i,j] = mv[j]/mv[i] * q01[i,j]

    q20 = numpy.zeros((nsp, nsp))
    for i in range(nsp):
        for j in range(nsp):
            q20[i,j] = (mv[j]/mv[i])**2 * q02[i,j]

    q21 = numpy.zeros((nsp, nsp))
    for i in range(nsp):
        for j in range(nsp):
            q21[i,j] = mv[j]/mv[i] * q12[i,j]

    q30 = numpy.zeros((nsp, nsp))
    for i in range(nsp):
        for j in range(nsp):
            q30[i,j] = (mv[j]/mv[i])**3 * q03[i,j]

    q31 = numpy.zeros((nsp, nsp))
    for i in range(nsp):
        for j in range(nsp):
            q31[i,j] = (mv[j]/mv[i])**2 * q13[i,j]

    q32 = numpy.zeros((nsp, nsp))
    for i in range(nsp):
        for j in range(nsp):
            q32[i,j] = mv[j]/mv[i] * q23[i,j]
    
    qq = numpy.zeros((4*nsp, 4*nsp))

    qq[0*nsp:1*nsp, 0*nsp:1*nsp] = q00
    qq[0*nsp:1*nsp, 1*nsp:2*nsp] = q01
    qq[0*nsp:1*nsp, 2*nsp:3*nsp] = q02
    qq[0*nsp:1*nsp, 3*nsp:4*nsp] = q03

    qq[1*nsp:2*nsp, 0*nsp:1*nsp] = q10
    qq[1*nsp:2*nsp, 1*nsp:2*nsp] = q11
    qq[1*nsp:2*nsp, 2*nsp:3*nsp] = q12
    qq[1*nsp:2*nsp, 3*nsp:4*nsp] = q13

    qq[2*nsp:3*nsp, 0*nsp:1*nsp] = q20
    qq[2*nsp:3*nsp, 1*nsp:2*nsp] = q21
    qq[2*nsp:3*nsp, 2*nsp:3*nsp] = q22
    qq[2*nsp:3*nsp, 3*nsp:4*nsp] = q23

    qq[3*nsp:4*nsp, 0*nsp:1*nsp] = q30
    qq[3*nsp:4*nsp, 1*nsp:2*nsp] = q31
    qq[3*nsp:4*nsp, 2*nsp:3*nsp] = q32
    qq[3*nsp:4*nsp, 3*nsp:4*nsp] = q33

    return qq

def qhat(mix):
    nsp = len(mix.species)
    nv = mix.calculate_composition()
    mv = numpy.array([sp.molarmass / kav for sp in mix.species])

    Q11 = Qij_mix(mix, 1, 1)
    Q12 = Qij_mix(mix, 1, 2)
    Q13 = Qij_mix(mix, 1, 3)
    Q22 = Qij_mix(mix, 2, 2)
    Q23 = Qij_mix(mix, 2, 3)
    Q24 = Qij_mix(mix, 2, 4)
    Q33 = Qij_mix(mix, 3, 3)

    qhat00 = numpy.zeros((nsp, nsp))
    for i in range(nsp):
        for j in range(nsp):
            sumval = 0
            for l in range(nsp):
                term1 = nv[l]*mv[l]**(1/2) / (mv[i] + mv[l])**(3/2)
                term2 = ((delta(i,j) - delta(j,l)) * 10/3*mv[j]*Q11[i,l] 
                        + (delta(i,j) + delta(j,l)) * 2*mv[l]*Q22[i,l])
                sumval += term1*term2
            qhat00[i,j] = 8*nv[i]*(mv[i]/mv[j]) * sumval
            
    qhat01 = numpy.zeros((nsp, nsp))
    for i in range(nsp):
        for j in range(nsp):
            sumval = 0
            for l in range(nsp):
                term1 = nv[l]*mv[l]**(3/2) / (mv[i] + mv[l])**(5/2)
                term2 = ((delta(i,j)-delta(j,l)) 
                         * mv[j]*(35/3*Q11[i,l] - 14*Q12[i,l]) 
                         + (delta(i,j) + delta(j,l)) 
                         * mv[l] * (7*Q22[i,l] - 8*Q23[i,l]))
                sumval += term1*term2
            qhat01[i,j] = 8*nv[i]*(mv[i]/mv[j])**2 * sumval
            
    qhat11 = numpy.zeros((nsp, nsp))
    for i in range(nsp):
        for j in range(nsp):
            sumval = 0
            for l in range(nsp):
                term1 = nv[l]*mv[l]**(1/2) / (mv[i] + mv[l])**(7/2)
                term2 = ((delta(i,j)-delta(j,l)) 
                         * mv[j]*(1/6*(140*mv[j]**2+245*mv[l]**2)*Q11[i,l] 
                                  - mv[l]**2*(98*Q12[i,l] - 64*Q13[i,l] 
                                              - 24*Q33[i,l])) 
                        + (delta(i,j)+delta(j,l)) * mv[l] 
                        * (1/6*(154*mv[j]**2+147*mv[l]**2)*Q22[i,l] 
                           - mv[l]**2*(56*Q23[i,l] - 40*Q24[i,l])))
                sumval += term1*term2
            qhat11[i,j] = 8*nv[i]*(mv[i]/mv[j])**2 * sumval

    qhat10 = numpy.zeros((nsp, nsp))
    for i in range(nsp):
        for j in range(nsp):
            qhat10[i,j] = mv[j]/mv[i] * qhat01[i,j]
    
    qq = numpy.zeros((2*nsp, 2*nsp))

    qq[0*nsp:1*nsp, 0*nsp:1*nsp] = qhat00
    qq[0*nsp:1*nsp, 1*nsp:2*nsp] = qhat01

    qq[1*nsp:2*nsp, 0*nsp:1*nsp] = qhat10
    qq[1*nsp:2*nsp, 1*nsp:2*nsp] = qhat11

    return qq

### Transport property calculations ############################################

def Dij(mix):
    ''' Diffusion coefficients, calculation per Devoto 1966 (eqns 3 and 6). 
    Fourth-order approximation.
    '''
    nsp = len(mix.species)
    nv = mix.calculate_composition()
    mv = numpy.array([sp.molarmass / kav for sp in mix.species])
    ntot = numpy.sum(nv)
    rho = mix.calculate_density()

    qq = q(mix)
    # qq = q(mix)[:nsp, :nsp]
    invq = numpy.linalg.inv(qq)
    dmat = numpy.zeros((nsp,nsp))
    bvec = numpy.zeros(4*nsp)
    # bvec = numpy.zeros(nsp)
    for i in range(nsp):
        for j in range(nsp):
            dij = numpy.array([delta(h,i) - delta(h,j) for h in range(0,nsp)])
            bvec[:nsp] = 3*numpy.sqrt(pi)*dij
            cflat = invq.dot(bvec)
            cip = cflat.reshape(4, nsp)
            # cip = cflat.reshape(1, nsp)
            dmat[i,j] = (rho * nv[i] / (2*ntot*mv[j]) 
                         * numpy.sqrt(2*kb*mix.T/mv[i]) * cip[0,i])
    
    return dmat

def DTi(mix):
    ''' Thermal diffusion coefficients, calculation per Devoto 1966 (eqns 4 
    and 5). Fourth-order approximation.
    '''
    nsp = len(mix.species)
    nv = mix.calculate_composition()
    mv = numpy.array([sp.molarmass / kav for sp in mix.species])

    qq = q(mix)
    invq = numpy.linalg.inv(qq)
    bvec = numpy.zeros(4*nsp)
    bvec[nsp:2*nsp] = -15/2*numpy.sqrt(pi)*nv
    aflat = invq.dot(bvec)
    aip = aflat.reshape(4, nsp)
 
    return 0.5 * nv * mv * numpy.sqrt(2*kb*mix.T/mv) * aip[0]

def viscosity(mix):
    ''' Viscosity, calculation per Devoto 1966 (eqns 19 and 20). Second-order 
        approximation.
    '''
    nsp = len(mix.species)
    nv = mix.calculate_composition()
    mv = numpy.array([sp.molarmass / kav for sp in mix.species])

    qq = qhat(mix)
    invq = numpy.linalg.inv(qq)
    bvec = numpy.zeros(2*nsp)
    bvec[:nsp] = 5*nv*numpy.sqrt(2*pi*mv/(kb*mix.T))
    bflat = invq.dot(bvec)
    bip = bflat.reshape(2, nsp)
    
    return 0.5*kb*mix.T*numpy.sum(nv*bip[0])

def electricalconductivity(mix):
    ''' Electrical conductivity, calculation per Devoto 1966 (eqn 29). This 
    simplification neglects heavy ion contributions to the current.
    '''
    nv = mix.calculate_composition()
    mv = numpy.array([sp.molarmass / kav for sp in mix.species])
    ntot = numpy.sum(nv)
    rho = mix.calculate_density()

    D1 = Dij(mix)[-1,:]

    sumval = 0
    for spj, D1j, mj, nj in zip(mix.species, D1, mv, nv):
        sumval+=nj*mj*spj.chargenumber*D1j
    premult = qe**2*ntot / (rho*kb*mix.T) 
    
    return premult * sumval

def thermalconductivity(mix, rel_delta_T=0.001, DTterms_yn=False):
    ''' Thermal conductivity, calculation per Devoto 1966 (eqn 2 and 19). 
    Numerical derivative performed to obtain dxi/dT for del(x) in the di 
    expression. Electrons are currently excluded from the thermal diffusion and 
    reactional component calculations due to inconsistencies in results.
    '''
    nsp = len(mix.species)
    nv = mix.calculate_composition()
    mv = numpy.array([sp.molarmass / kav for sp in mix.species])
    ntot = numpy.sum(nv)
    hv = mix.calculate_species_enthalpies()
    rho = mix.calculate_density()
    
    ### translational tk components ###
    qq = q(mix)
    invq = numpy.linalg.inv(qq)
    bvec = numpy.zeros(4*nsp)
    bvec[nsp:2*nsp] = -15/2*numpy.sqrt(pi)*nv
    aflat = invq.dot(bvec)
    aip = aflat.reshape(4, nsp)
    kdash = -5/4 * kb  * numpy.sum(nv * numpy.sqrt(2*kb*mix.T/mv) * aip[1])

    ### thermal diffusion tk components ###
    if DTterms_yn:
        locDTi = DTi(mix)
        kdt = numpy.sum(hv[:-1]*locDTi[:-1] / mix.T)
    else:
        kdt = 0

    ### reactional tk components ###
    locDij = Dij(mix)
    Tval = mix.T
    mix.T = Tval * (1 + rel_delta_T)
    nvpos = mix.calculate_composition()
    mix.T = Tval * (1 - rel_delta_T)
    nvneg = mix.calculate_composition()
    mix.T = Tval
    xvpos = nvpos / numpy.sum(nvpos)
    xvneg = nvneg / numpy.sum(nvneg)
    dxdT = (xvpos - xvneg) / (2 * rel_delta_T * mix.T)
    krxn_enth = 0
    for j in range(nsp-1):
        sumi = 0
        for i in range(nsp-1):
            sumi += hv[i]*mv[i]*locDij[i,j]
        krxn_enth += -ntot**2 / rho * mv[j] * sumi * dxdT[j]

    if DTterms_yn:
        krxn_therm = ntot * kb * Tval * numpy.sum(locDTi[:-1] * dxdT[:-1] 
                                                  / (nv[:-1] * mv[:-1]))
    else:
        krxn_therm = 0
    
    return kdash + kdt + krxn_enth + krxn_therm
