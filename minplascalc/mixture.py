import numpy
import logging
import warnings
from scipy import constants
from . import species as _species
from . import functions_transport, functions_radiation

__all__ = ['lte_from_names', 'LTE']


def lte_from_names(names, x0, T, P):
    """Create a LTE mixture from a list of species names using the species 
    database.

    Parameters
    ----------
    names : list of str
        Names of the species.
    x0 : list of float
        Initial value of mole fractions for each species, typically the 
        room-temperature composition of the plasma-generating gas.
    T : float
        LTE plasma temperature, in K.
    P : float
        LTE plasma pressure, in Pa.
        
    Returns
    -------
    An LTE object instance.
    """
    if 'e' in names:
        raise ValueError('Electrons are added automatically, please don\'t '
                         'include them in your species list.')
    species = [_species.from_name(nm) for nm in names]
    return LTE(species, x0, T, P, 1e20, 1e-10, 1000)
    

class LTE:
    def __init__(self, species, x0, T, P, gfe_ni0, gfe_reltol, gfe_maxiter):
        """Class representing a thermal plasma specification with multiple
        species, and methods for calculating equilibrium species concentrations
        at different temperatures and pressures using the principle of Gibbs
        free energy minimisation.

        Parameters
        ----------
        species : list or tuple of obj
            All species participating in the mixture (excluding electrons which
            are added automatically), as minplascalc Species objects.
        x0 : list or tuple of float
            Constraint mole fractions for each species, typically the 
            room-temperature composition of the plasma-generating gas.
        T : float
            LTE plasma temperature, in K.
        P : float
            LTE plasma pressure, in Pa.           
        gfe_ni0 : float
            Gibbs Free Energy minimiser solution control: Starting estimate for 
            number of particles of each species. Typically O(1e20).         
        gfe_reltol : float
            Gibbs Free Energy minimiser solution control: Relative tolerance at
            which solution for particle numbers is considered converged. 
            Typically O(1e-10).
        gfe_maxiter : int
            Gibbs Free Energy minimiser solution control: Bailout loop count 
            value for iterative solver. Typically O(1e3).
        """
        if 'e' in [sp.name for sp in species]:
            raise ValueError('Electrons are added automatically, please don\'t '
                             'include them in your species list.')
        if len(species) != len(x0):
            raise ValueError('Lists species and x0 must be the same length.')
        self.__species = tuple(list(species) + [_species.Electron()])
        self.x0 = x0
        self.T = T
        self.P = P
        self.gfe_ni0 = gfe_ni0
        self.gfe_reltol = gfe_reltol
        self.gfe_maxiter = gfe_maxiter
        self.__isLTE = False
    
    @property
    def species(self):
        return self.__species
    
    @species.setter
    def species(self, species):
        raise TypeError('Attribute species is read-only. Please create a new '
                        'Mixture object if you wish to change the plasma '
                        'species.')

    @property
    def x0(self):
        return self.__x0
    
    @x0.setter
    def x0(self, x0):
        if len(x0) == len(self.species)-1:
            self.__isLTE = False
            self.__x0 = tuple(list(x0) + [0])
        else:
            raise ValueError('Please specify constraint mole fractions for all '
                             'species except electrons.')
        
    @property
    def T(self):
        return self.__T
    
    @T.setter
    def T(self, T):
        self.__isLTE = False
        self.__T = T
        
    @property
    def P(self):
        return self.__P
    
    @P.setter
    def P(self, P):
        self.__isLTE = False
        self.__P = P
    
    def __repr__(self):
        return (f'{self.__class__.__name__}(species={self.species},'
                f'x0={self.x0},T={self.T},P={self.P},gfe_ni0={self.gfe_ni0},'
                f'gfe_reltol={self.gfe_reltol},gfe_maxiter={self.gfe_maxiter})')

    def __str__(self):
        return (f'LTE mixture species: '
                f'{tuple([sp.name for sp in self.species[:-1]])}\n'
                f'Initial composition: {self.x0[:-1]}\n'
                f'Temperature: {self.T} K\nPressure: {self.P} Pa')
        
    def __recalcE0i(self):
        """Calculate the reference energy values for all species, including 
        ionisation energy lowering from limitation theory of Stewart & 
        Pyatt 1966 (lowering only applied to positive ions).

        Returns
        -------
        tuple(ndarray, ndarray)
            Reference energy and ionisation energy lowering of each species 
            in the mixture, in J.
        """
        nspecies = len(self.species)
        kbt = constants.Boltzmann * self.T
        ndi = self.__ni * self.P / (self.__ni.sum() * kbt) 
        E0, dE = numpy.zeros(nspecies), numpy.zeros(nspecies)
        
        for i, sp in enumerate(self.species):
            if sum(dv for kv, dv in sp.stoichiometry.items()) >= 2:
                E0[i] = -sp.dissociationenergy
        weightedchargesumsqd, weightedchargesum = 0, 0
        for sp, nd in zip(self.species, ndi):
            if sp.chargenumber > 0:
                weightedchargesum += nd * sp.chargenumber
                weightedchargesumsqd += nd * sp.chargenumber ** 2
        zstar = weightedchargesumsqd / weightedchargesum
        debyed3 = (constants.epsilon_0 * kbt / (4 * numpy.pi * (zstar + 1) 
                   * ndi[-1] * constants.elementary_charge ** 2)) ** (3/2)
        for i, sp in enumerate(self.species):
            if sp.chargenumber > 0:
                ai3 = 3 * sp.chargenumber / (4 * numpy.pi * ndi[-1])
                dE[i] = kbt * ((ai3/debyed3 + 1)**(2/3) - 1) / (2 * (zstar + 1))
        neutralsp = [sp for sp in self.species if sp.chargenumber==0]
        for nsp in neutralsp:
            ncsp = [(i, sp) for i, sp in enumerate(self.species) 
                    if (sp.stoichiometry==nsp.stoichiometry 
                        and sp.chargenumber <= 0)]
            ncsp.sort(key=lambda sp: sp[1].chargenumber, reverse=True)
            pcsp = [(i, sp) for i, sp in enumerate(self.species) 
                    if (sp.stoichiometry==nsp.stoichiometry 
                        and sp.chargenumber >= 0)]
            pcsp.sort(key=lambda sp: sp[1].chargenumber, reverse=False)             
            for (ifrom, spfrom), (ito, spto) in zip(pcsp[:-1], pcsp[1:]):
                E0[ito] = (E0[ifrom] + spfrom.ionisationenergy - dE[ifrom])
            for (ifrom, spfrom), (ito, spto) in zip(ncsp[:-1], ncsp[1:]):
                E0[ito] = (E0[ifrom] - spto.ionisationenergy + dE[ito])
        return E0, dE

    def calculate_composition(self):
        """Calculate the LTE composition of the plasma in particles/m3.
        
        Returns
        -------
        ndarray
            Number density of each species in the plasma as listed in 
            Mixture.species, in particles/m3.
        """
        nspecies = len(self.species)
        kbt = constants.Boltzmann*self.T
        
        if not self.__isLTE:
            elements = [{'name': nm, 'stoichcoeff': None, 'ntot': 0}
                        for nm in sorted(set(s for sp in self.species
                                             for s in sp.stoichiometry))]
            for elm in elements:
                elm['stoichcoeff'] = [sp.stoichiometry.get(elm['name'], 0)
                                      for sp in self.species]
            for elm in elements:
                elm['ntot'] = sum(1e24 * c * x0loc
                                  for c, x0loc in zip(elm['stoichcoeff'], 
                                                      self.x0))
            minimiser_dof = nspecies + len(elements) + 1
            gfematrix = numpy.zeros((minimiser_dof, minimiser_dof))
            gfevector = numpy.zeros(minimiser_dof)
            for i, elm in enumerate(elements):
                gfevector[nspecies + i] = elm['ntot']
                for j, sc in enumerate(elm['stoichcoeff']):
                    gfematrix[nspecies + i, j] = sc
                    gfematrix[j, nspecies + i] = sc
            for j, qc in enumerate(sp.chargenumber for sp in self.species):
                gfematrix[-1, j] = qc
                gfematrix[j, -1] = qc
    
            self.__ni = numpy.full(nspecies, self.gfe_ni0)
            governorfactors = numpy.linspace(0.9, 0.1, 9)
            successyn = False
            governoriters = 0
            while not successyn and governoriters < len(governorfactors):
                successyn = True
                governorfactor = governorfactors[governoriters]
                reltol = self.gfe_reltol * 10
                minimiseriters = 0
                while reltol > self.gfe_reltol:
                    self.__E0, self.__dE = self.__recalcE0i()
                    nisum = self.__ni.sum()
                    V = nisum * kbt / self.P
                    offdiag, ondiag = -kbt/nisum, numpy.diag(kbt/self.__ni)
                    gfematrix[:nspecies, :nspecies] = offdiag + ondiag
                    total = [sp.partitionfunction_total(V, self.T, dE) 
                             for sp, dE in zip(self.species, self.__dE)]
                    mu = -kbt * numpy.log(total/self.__ni) + self.__E0
                    gfevector[:nspecies] = -mu
                    solution = numpy.linalg.solve(gfematrix, gfevector)
                    newni = solution[0:nspecies]
                    deltani = abs(newni - self.__ni)
                    maxniindex = newni.argmax()
                    reltol = deltani[maxniindex] / solution[maxniindex]
    
                    maxalloweddeltani = governorfactor * self.__ni
                    deltani = deltani.clip(min=maxalloweddeltani)
                    newrelaxfactors = maxalloweddeltani / deltani
                    relaxfactor = newrelaxfactors.min()
                    self.__ni = (1-relaxfactor)*self.__ni + relaxfactor*newni
    
                    minimiseriters += 1
                    if minimiseriters > self.gfe_maxiter:
                        successyn = False
                        break 
                governoriters += 1
            if not successyn:
                warnings.warn('Minimiser could not find a converged solution, '
                              'results may be inaccurate.')
            # noinspection PyUnboundLocalVariable
            logging.debug(governoriters, relaxfactor, reltol)
            logging.debug(self.__ni)

        self.__isLTE = True
        return self.__ni*self.P / (self.__ni.sum()*kbt) 

    def calculate_density(self):
        """Calculate the LTE density of the plasma.

        Returns
        -------
        float
            Fluid density, in kg/m3.
        """
        ndi = self.calculate_composition()
        return sum(nd * sp.molarmass / constants.Avogadro
                   for sp, nd in zip(self.species, ndi))

    def calculate_species_enthalpies(self):
        """Calculate the LTE enthalpy for each component in the plasma. These
        are needed for calculation of the effective thermal conductivity.
        
        Returns
        -------
        list of floats
            Enthalpies of each species, in J/kg.
        """
        enthalpies = [(sp.internal_energy(self.T, dE) 
                       + E0
                       + constants.Boltzmann * self.T 
                       )
                      for sp, dE, E0 in zip(self.species, self.__dE, self.__E0)]
        molmasses = [sp.molarmass / constants.Avogadro for sp in self.species]
        return numpy.array(enthalpies) / numpy.array(molmasses)
        #return constants.Avogadro * numpy.array(enthalpies)

    def calculate_enthalpy(self):
        """Calculate the LTE enthalpy of the plasma. Referenced to zero at zero
        Kelvin.

        Returns
        -------
        float
            Enthalpy, in J/kg.
        """
        ndi = self.calculate_composition()
        imin = numpy.argmin(self.__E0)
        h0 = self.__E0[imin] / self.species[imin].molarmass
        weightedenthalpy = sum(constants.Avogadro * nd 
                               * (sp.internal_energy(self.T, dE) + E0 
                                  + constants.Boltzmann * self.T 
                                  - h0*sp.molarmass) 
                               for sp, nd, dE, E0 in zip(self.species, ndi, 
                                                         self.__dE, self.__E0))
        weightedmolmass = sum(nd * sp.molarmass
                              for sp, nd in zip(self.species, ndi))
        return weightedenthalpy / weightedmolmass

    def calculate_heat_capacity(self, rel_delta_T=0.001):
        """Calculate the LTE heat capacity at constant pressure of the plasma 
        based on current conditions and species composition. 
        
        This is done by performing multiple LTE composition recalculations and 
        can be time-consuming to execute - when performing large quantities of 
        Cp calculations at different temperatures, it is more efficient to 
        calculate enthalpies and perform a numerical derivative external to 
        minplascalc.

        Returns
        -------
        float
            Heat capacity, in J/kg.K.
        """
        startT = self.T
        self.T = startT * (1-rel_delta_T)        
        enthalpylow = self.calculate_enthalpy()
        self.T = startT * (1+rel_delta_T)        
        enthalpyhigh = self.calculate_enthalpy()
        self.T = startT
        return (enthalpyhigh - enthalpylow) / (2 * rel_delta_T * self.T)

    def calculate_viscosity(self):
        """Calculate the LTE viscosity of the plasma in Pa.s based on current
        conditions and species composition.
        """
        return functions_transport.viscosity(self)

    def calculate_thermal_conductivity(self, rel_delta_T=0.001, 
                                       DTterms_yn=True, ni_limit=1e8):
        """Calculate the LTE thermal conductivity of the plasma in W/m.K.
        """
        return functions_transport.thermalconductivity(self, rel_delta_T, 
                                                       DTterms_yn, ni_limit)

    def calculate_electrical_conductivity(self):
        """Calculate the LTE electrical conductivity of the plasma in 1/ohm.m.
        """
        return functions_transport.electricalconductivity(self)

    def calculate_total_emission_coefficient(self):
        """Calculate the LTE total radiation emission coefficient of the plasma 
        in W/m3.sr.
        """
        return functions_radiation.total_emission_coefficient(self)
    