module minplas_collision_kernel
  use, intrinsic :: iso_c_binding
  implicit none

  real(c_double), parameter :: pi = 3.1415926535897932384626433832795d0
  real(c_double), parameter :: ke = 8.987551786170797d9
  real(c_double), parameter :: electron_charge = 1.602176634d-19
  real(c_double), parameter :: boltzmann = 1.380649d-23
  real(c_double), parameter :: electron_mass = 9.1093837139d-31
  real(c_double), parameter :: hbar = 1.0545718176461565d-34
  real(c_double), parameter :: gas_constant = 8.31446261815324d0
  real(c_double), parameter :: kelvin_to_ev = 8.617333262145179d-5
  real(c_double), parameter :: euler_gamma = 0.5772156649015329d0

contains

  pure real(c_double) function omega_star(x, a, recursion, base_s) result(value)
    real(c_double), intent(in) :: x, a(7)
    integer(c_int), intent(in) :: recursion, base_s
    real(c_double) :: k1, k2, c, sigmoid1, sigmoid2, g
    real(c_double) :: d1, d2, g1, g2, polynomial

    k1 = 2.0d0 / a(4)
    k2 = 2.0d0 / a(7)
    c = a(1) + a(2) * x
    sigmoid1 = 1.0d0 / (1.0d0 + exp(-k1 * (x - a(3))))
    sigmoid2 = 1.0d0 / (1.0d0 + exp(-k2 * (x - a(6))))
    g = c * sigmoid1 + a(5) * sigmoid2

    if (recursion == 0) then
      value = exp(g)
      return
    end if

    d1 = sigmoid1 * (1.0d0 - sigmoid1)
    d2 = sigmoid2 * (1.0d0 - sigmoid2)
    g1 = a(2) * sigmoid1 + c * k1 * d1 + a(5) * k2 * d2
    polynomial = 1.0d0 + g1 / real(base_s + 2, c_double)
    if (recursion == 1) then
      value = exp(g) * polynomial
      return
    end if

    g2 = 2.0d0 * a(2) * k1 * d1 &
      + c * k1**2 * d1 * (1.0d0 - 2.0d0 * sigmoid1) &
      + a(5) * k2**2 * d2 * (1.0d0 - 2.0d0 * sigmoid2)
    polynomial = polynomial &
      + (g1 * polynomial + g2 / real(base_s + 2, c_double)) &
      / real(base_s + 3, c_double)
    value = exp(g) * polynomial
  end function omega_star


  pure real(c_double) function omega_fit( &
      table, sigma, epsilon_0, beta_values, l, s, temperature) result(value)
    real(c_double), intent(in) :: table(*), sigma, epsilon_0
    real(c_double), intent(in) :: beta_values(3), temperature
    integer(c_int), intent(in) :: l, s
    integer(c_int), parameter :: base_orders(4) = [5, 4, 3, 4]
    integer(c_int) :: base_s, selected_s, coefficient, beta_index, table_index
    real(c_double) :: a(7), x

    base_s = base_orders(l)
    selected_s = min(s, base_s)
    do coefficient = 1, 7
      a(coefficient) = 0.0d0
      do beta_index = 1, 3
        ! C layout for table(4, 5, 7, 3) supplied by NumPy.
        table_index = (((l - 1) * 5 + selected_s - 1) * 7 &
          + coefficient - 1) * 3 + beta_index
        a(coefficient) = a(coefficient) &
          + table(table_index) * beta_values(beta_index)
      end do
    end do

    x = log(temperature * kelvin_to_ev / epsilon_0)
    value = omega_star(x, a, max(0, s - base_s), base_s) &
      * pi * sigma**2 * 1.0d-20
  end function omega_fit


  pure real(c_double) function coulomb_logarithm( &
      i, j, density_i, density_j, temperature, charges, electron_index) &
      result(value)
    integer(c_int), intent(in) :: i, j, charges(*), electron_index
    real(c_double), intent(in) :: density_i, density_j, temperature
    real(c_double) :: temperature_ev, electron_density_cgs
    real(c_double) :: density_i_cgs, density_j_cgs
    integer(c_int) :: charge_i, charge_j

    temperature_ev = temperature * kelvin_to_ev
    if (i == electron_index .and. j == electron_index) then
      electron_density_cgs = density_i * 1.0d-6
      value = 23.5d0 &
        - log(sqrt(electron_density_cgs) * temperature_ev**(-1.25d0)) &
        - sqrt(1.0d-5 + (log(temperature_ev) - 2.0d0)**2 / 16.0d0)
    else if (i == electron_index) then
      electron_density_cgs = density_i * 1.0d-6
      value = 23.0d0 - log(sqrt(electron_density_cgs) &
        * abs(charges(j)) * temperature_ev**(-1.5d0))
    else if (j == electron_index) then
      electron_density_cgs = density_j * 1.0d-6
      value = 23.0d0 - log(sqrt(electron_density_cgs) &
        * abs(charges(i)) * temperature_ev**(-1.5d0))
    else
      density_i_cgs = density_i * 1.0d-6
      density_j_cgs = density_j * 1.0d-6
      charge_i = charges(i)
      charge_j = charges(j)
      value = 23.0d0 - log(abs(charge_i * charge_j) / temperature_ev &
        * sqrt(density_i_cgs * abs(charge_i)**2 / temperature_ev &
        + density_j_cgs * abs(charge_j)**2 / temperature_ev))
    end if
  end function coulomb_logarithm


  subroutine collision_integrals_fortran( &
      species_count, moment_count, temperature, electron_index, &
      number_densities, moments, kinds, charges, fit_parameters, &
      electron_parameters, electron_gamma_ratios, resonant, &
      resonant_parameters, neutral_table, ion_table, psi_values, &
      sum1_values, sum2_values, values) bind(C)
    integer(c_int), value, intent(in) :: species_count, moment_count
    real(c_double), value, intent(in) :: temperature
    integer(c_int), value, intent(in) :: electron_index
    real(c_double), intent(in) :: number_densities(*)
    integer(c_int), intent(in) :: moments(*), kinds(*), charges(*)
    real(c_double), intent(in) :: fit_parameters(*)
    real(c_double), intent(in) :: electron_parameters(*)
    real(c_double), intent(in) :: electron_gamma_ratios(*)
    integer(c_int), intent(in) :: resonant(*)
    real(c_double), intent(in) :: resonant_parameters(*)
    real(c_double), intent(in) :: neutral_table(*), ion_table(*)
    real(c_double), intent(in) :: psi_values(*), sum1_values(*), sum2_values(*)
    real(c_double), intent(out) :: values(*)

    integer(c_int) :: i, j, k, l, s, kind, neutral
    integer(c_int) :: pair_index, moment_index, result_index, parameter_index
    real(c_double) :: b0_squared, collision_log, sigma, epsilon_0
    real(c_double) :: d1_parameter, d2_parameter, d3_parameter, d4_parameter
    real(c_double) :: tau, tau_power, tau_squared, gamma_argument
    real(c_double) :: a, b, molar_mass, log_term, zeta1, zeta2, cterm
    real(c_double) :: beta_values(3)
    real(c_double), parameter :: qc_c1(4) = [4.0d0, 12.0d0, 12.0d0, 16.0d0]
    real(c_double), parameter :: qc_c2(4) = [0.5d0, 1.0d0, &
      1.1666666666666666667d0, 1.3333333333333333333d0]

    do i = 1, species_count
      do j = 1, species_count
        pair_index = (i - 1) * species_count + j
        kind = kinds(pair_index)

        if (kind == 0) then
          b0_squared = (ke * charges(i) * charges(j) * electron_charge**2 &
            / (2.0d0 * boltzmann * temperature))**2
          collision_log = coulomb_logarithm(i, j, number_densities(i), &
            number_densities(j), temperature, charges, electron_index) &
            + log(2.0d0)
          do k = 1, moment_count
            moment_index = (k - 1) * 2
            l = moments(moment_index + 1)
            s = moments(moment_index + 2)
            result_index = (k - 1) * species_count**2 + pair_index
            values(result_index) = qc_c1(l) * pi / real(s * (s + 1), c_double) &
              * b0_squared * (collision_log - qc_c2(l) - 2.0d0 * euler_gamma &
              + psi_values(s + 1))
          end do
          cycle
        end if

        if (kind == 1) then
          if (i == electron_index) then
            neutral = j
          else
            neutral = i
          end if
          parameter_index = (neutral - 1) * 4
          d1_parameter = electron_parameters(parameter_index + 1)
          d2_parameter = electron_parameters(parameter_index + 2)
          d3_parameter = electron_parameters(parameter_index + 3)
          d4_parameter = electron_parameters(parameter_index + 4)
          tau = sqrt(2.0d0 * electron_mass * boltzmann * temperature) / hbar
          tau_power = tau**d3_parameter
          tau_squared = d4_parameter * tau**2 + 1.0d0
          do k = 1, moment_count
            s = moments((k - 1) * 2 + 2)
            gamma_argument = d3_parameter / 2.0d0 + s + 2.0d0
            result_index = (k - 1) * species_count**2 + pair_index
            values(result_index) = d1_parameter + d2_parameter * tau_power &
              * electron_gamma_ratios((neutral - 1) * 8 + s + 1) &
              / tau_squared**gamma_argument
          end do
          cycle
        end if

        parameter_index = (pair_index - 1) * 5
        sigma = fit_parameters(parameter_index + 1)
        epsilon_0 = fit_parameters(parameter_index + 2)
        beta_values = fit_parameters(parameter_index + 3:parameter_index + 5)
        do k = 1, moment_count
          moment_index = (k - 1) * 2
          l = moments(moment_index + 1)
          s = moments(moment_index + 2)
          result_index = (k - 1) * species_count**2 + pair_index
          if (kind == 2) then
            values(result_index) = omega_fit(neutral_table, sigma, epsilon_0, &
              beta_values, l, s, temperature)
          else if (resonant(pair_index) /= 0 .and. mod(l, 2) == 1) then
            parameter_index = (pair_index - 1) * 3
            a = resonant_parameters(parameter_index + 1)
            b = resonant_parameters(parameter_index + 2)
            molar_mass = resonant_parameters(parameter_index + 3)
            log_term = log(4.0d0 * gas_constant * temperature / molar_mass)
            zeta1 = sum1_values(s + 1)
            zeta2 = sum2_values(s + 1)
            cterm = pi**2 / 6.0d0 - zeta2 + zeta1**2
            values(result_index) = a**2 - zeta1 * a * b &
              + (b / 2.0d0)**2 * cterm + (b / 2.0d0)**2 * log_term**2 &
              + (zeta1 * b**2 / 2.0d0 - a * b) * log_term
          else
            values(result_index) = omega_fit(ion_table, sigma, epsilon_0, &
              beta_values, l, s, temperature)
          end if
        end do
      end do
    end do
  end subroutine collision_integrals_fortran

end module minplas_collision_kernel
