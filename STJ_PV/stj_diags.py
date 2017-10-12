#-*- coding: utf-8 -*-
"""
Module containing classes for diagnoistic variable calculation and diagnoistic plotting.
"""
import input_data
import stj_metric
import run_stj

class DiagMetrics(object):
    """
    Calculate diagnostic metrics about subtropical jet properties.
    """

    def __init__(self, stj_props, metric):
        """
        Setup DiagMetrics, input from a STJMetric and STJProperties classes.
        """
        self.props = stj_props
        self.metric = metric

    def jet_intensity(self):
        """
        Calculate jet intensity at identified points.
        """

    def annual_correlations(self):
        """
        Calculate annual correlations.
        """

    def monthly_correlations(self):
        """
        Calculate monthly correlations.
        """

    def seasonal_correlations(self):
        """
        Calculate seasonal correlations.
        """

    def polyfit_near_mean(self):
        """
        Perform a polynomial fit of pv contour near "expected" jet location mean.
        """

    def validate_near_mean(self):
        """
        Check that identified STJ position is near "expected" jet location mean.
        """

    def get_uwind_strength(self):
        """
        Calculate zonal wind profile.
        """
        # created named tuples to manage storage of index
        metric = collections.namedtuple('metric', 'name hemisphere intensity position')
        time_len = self.ipv_data['ipv'].shape[0]
        STJ_int = np.zeros(time_len)
        STJ_pos = np.zeros(time_len)
        pdb.set_trace()

        for hemi in ['NH', 'SH']:
            for time_loop in range(time_len):
                # for time_loop in xrange(1):

                if hemi == 'NH':
                    lat = self.lat_NH
                    STJ_phi = self.NH_STJ_phi[time_loop]
                    STJ_th = self.NH_STJ_theta[time_loop]
                else:
                    lat = self.lat_SH
                    STJ_phi = self.SH_STJ_phi[time_loop]
                    STJ_th = self.SH_STJ_theta[time_loop]

                # step 8. interpolate u wind
                u_zonal = MeanOverDim(data=self.u_th[time_loop, :, :, :], dim=2)
                u_zonal_function = interpolate.interp2d(
                    self.lat, self.theta_lev, u_zonal, kind='cubic')
                u_zonal_interp = u_zonal_function(lat, self.theta_interp)

                # step 9: for the 2.0 max derivative latitude find the uwind strength
                # get element closest to phi and theta points

                elem_phi = FindClosestElem(STJ_phi, lat)[0]
                elem_theta = FindClosestElem(STJ_th, self.theta_interp)[0]

                STJ_int[time_loop] = u_zonal_interp[elem_theta, elem_phi]
                STJ_pos[time_loop] = STJ_phi

            if hemi == 'NH':
                Metric_NH = metric(name='STJ', hemisphere=hemi,
                                   intensity=STJ_int, position=STJ_pos)
            else:
                Metric_SH = metric(name='STJ', hemisphere=hemi,
                                   intensity=STJ_int, position=STJ_pos)

        return Metric_NH, Metric_SH

class DiagPlots(object):
    """
    Plot diagnostic metrics about subtropical jet properties.
    """

    def __init__(self, stj_props, metric):
        """
        Setup DiagMetrics, input from STJIPVMetric and STJProperties classes.
        """
        self.props = stj_props
        self.metric = metric

    def test_method_plot(self, date):
        """
        Plot individual components of the metric to ensure it is working properly.

        Parmaeters
        ----------
        date : A :py:meth:`~datetime.datetime` instance to select and plot

        """
        data = input_data.InputData(stj_props, date_s=date, date_e=date)

    def compare_fd_cby(self):
        """
        Plot comparison of finite difference and Chebyshev polynomial derivative jet pos.
        """

    def test_second_derr(self):
        """
        Plot test of second derivative of PV contour.
        """

    def jet_lat_timeseries(self):
        """
        Plot jet location over time for both NH and SH.
        """

    def partial_correlation_matrix(self):
        """
        Plot matrix of partial correlations.
        """
