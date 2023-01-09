import os
if os.environ.get('RUBIN_SIM_DATA_DIR') is None:
    os.environ["RUBIN_SIM_DATA_DIR"] = "/global/cfs/cdirs/lsst/groups/PZ/RUBIN_SIM_DATA"
# import rubin_sim
from rubin_sim import maf
import numpy as np
import healpy
# import matplotlib.pyplot as plt
import photerr
import pandas as pd
from rail.estimation.algos.simpleSOM import Inform_SimpleSOMSummarizer, SimpleSOMSummarizer
from rail.estimation.algos.knnpz import Inform_KNearNeighPDF, KNearNeighPDF
from rail.core.stage import RailStage
from rail.core.data import TableHandle
from rail.evaluation.metrics.pointestimates import PointStatsEz, PointSigmaIQR, PointBias, PointOutlierRate


def pixel_from_radec(ra, dec, nside=64):
    theta = np.pi / 180. * (90. - dec)
    phi = np.pi / 180. * ra
    pix = healpy.ang2pix(nside, theta, phi)
    return pix


class PZExgalDepths(maf.metrics.BaseMetric):

    """
    Calculate the extragalactic coadded depth in each filter.
    Set HEALpix to badval (i.e., disclude regions with):
     MW dust extinction E(B-V) > 0.2
     number of filters < nfilters_limit
     i-band depth < i_lim_mag
    Such HEALpix would likely not be included in cosmological analyses.
    """

    def __init__(self, m5Col='fiveSigmaDepth', units='mag', maps=['DustMap'],
                 wavelen_min=None, wavelen_max=None, wavelen_step=1., filterCol='filter',
                 nfilters_limit=6, implement_depth_ebv_cut=False, i_lim_mag=26.0, **kwargs):
        self.filternames = ['u', 'g', 'r', 'i', 'z', 'y']
        self.m5Col = m5Col
        self.filterCol = filterCol
        self.nfilters_limit = int(nfilters_limit)
        self.implement_depth_ebv_cut = implement_depth_ebv_cut
        self.i_lim_mag = i_lim_mag
        self.coaddSimple = maf.Coaddm5Metric()

        ### set up for i-band extincted coadded depth
        if self.implement_depth_ebv_cut:
            self.coadd_iband_with_dust = maf.ExgalM5(lsstFilter='i', m5Col=self.m5Col, units=units, **kwargs)

        super(PZExgalDepths, self).__init__(col=[self.m5Col, self.filterCol], maps=maps, units=units,
                                            **kwargs)
        self.metricDtype = 'object'

    def run(self, dataslice, slicePoint=None):

        # First, find the coadd depths in each healpix pixel
        coadd_depths = []
        nfilters = 0
        for filtername in self.filternames:
            in_filt = np.where(dataslice[self.filterCol] == filtername)[0]
            if len(in_filt) > 0:
                coadd_depths.append(self.coaddSimple.run(dataslice[in_filt]))
                nfilters += 1
            else:
                coadd_depths.append(self.badval)

        if self.implement_depth_ebv_cut:
            ### find the i-band extincted coadded depth
            ext_iband_coadd = self.coadd_iband_with_dust.run(dataSlice=dataslice,
                                                             slicePoint=slicePoint)
            ### get ebv
            ebv = slicePoint['ebv']

        ### figure out the conditions with the number of filters in which we want coverage
        ###  a field considered for cosmology will have coverage in all 6 filters
        if self.nfilters_limit == 6:
            discard_condition = (nfilters != 6)
        else:
            discard_condition = (nfilters <= self.nfilters_limit)

        ### now incorporaate depth + ebv cuts if needed
        if self.implement_depth_ebv_cut:
            discard_condition = discard_condition or (ebv > 0.2) or (ext_iband_coadd < self.i_lim_mag)

        ### mask the data point if dicard_condition is true
        if discard_condition:
            ### return a single badval which will mask the datapoint in the bundle.metricValues.
            coadd_depths = self.badval

        return coadd_depths


class PZDDFBinsMetric(object):

    def __init__(self, coadd_depths, bands=None, surveylist=None, filedict=None, surveyradec=None):
        """init funciton for the bins metric
        takes in the following
        Params
        ------
        coadd_depths (array): array of floats with the m5 values in each band
        bands (array): array of strings with the band names
        surveylist (list): list of strings with the names of the specz surveys
        filedict (dict): dictionary containing the same surveys of surveylist as keys
          and the file paths to the parquet files of the mock data for each survey as
          the values
        surveradec (dict): dictionary containing the same surveys of surveylist as keys
          and ra,dec pairs for each survey as the values
        """
        default_filedict = {'cosmos': "/global/cfs/cdirs/lsst/groups/PZ/users/sschmidt/DDFSTUFF/MOCK_COSMOS.pq",
                            'deep2': "/global/cfs/cdirs/lsst/groups/PZ/users/sschmidt/DDFSTUFF/MOCK_DEEP2.pq",
                            'vvds': "/global/cfs/cdirs/lsst/groups/PZ/users/sschmidt/DDFSTUFF/MOCK_VVDS.pq"
                            }
        default_radecdict = {'cosmos': [150.1, 2.18], 'deep2': [352.5, 0.0], 'vvds': [36.5, -4.5]}

        # set some defaults
        if bands is None:
            bands = ['u', 'g', 'r', 'i', 'z', 'y']
            print(f"using defauilt bands {bands}")
        if surveylist is None:
            print(f"using default list {surveylist}")
            surveylist = ['cosmos', 'deep2', 'vvds']
        if filedict is None:
            print(f"using default filedict {filedict}")
            filedict = default_filedict
        if surveyradec is None:
            print(f"using default ra decs {default_radecdict}")
            radecdict = default_radecdict
        self.coadd_depths = coadd_depths
        self.bands = bands
        self.filternames = bands
        self.surveylist = surveylist
        self.filedict = filedict
        self.radecdict = radecdict

    def run(self, train_file, test_file, Nbins):

        if not isinstance(Nbins, int):
            raise ValueError("Nbins must be an integer")

        DS = RailStage.data_store
        DS.__class__.allow_overwrite = True

        # Make the training file
        # train_file = self.make_training_file(coadd_depths)

        # train SOM with training data

        # add data to datastore
        test_data = DS.add_data("test_data", test_file, TableHandle)

        maglims = dict(u=27.79, g=29.04, r=29.06, i=28.62, z=27.98, y=27.05)
        som_dict = dict(usecols=self.bands, ref_column_name='i', mag_limits=maglims,
                        som_sigma=9.0, model="SOM_model.pkl", seed=87,
                        m_dim=41, n_dim=41, hdf5_groupname="", alias="somModel")
        train_som = Inform_SimpleSOMSummarizer.make_stage(name="train_SOM", **som_dict)
        train_som.inform(train_file)

        # train knnpz as well
        knndict = dict(column_names=self.bands, ref_column_name='i',
                       model="KNN_model.pkl", hdf5_groupname="",
                       alias="knnModel", mag_limits=maglims)
        train_knn = Inform_KNearNeighPDF.make_stage(name="train_knn", **knndict)
        train_knn.inform(train_file)
        testknn_dict = dict(column_names=self.bands, ref_column_name='i',
                            model=train_knn.get_handle("model"),
                            alias="knntest", mag_limits=maglims,
                            hdf5_groupname='', chunk_size=5000)

        test_knn = KNearNeighPDF.make_stage(name="estimate_knn", **testknn_dict)

        knnens = test_knn.estimate(test_data)
        self.knnpdfs = knnens

        # calculate global point estimate metric
        zb = knnens.data.ancil['zmode'].flatten()
        truez = np.array(test_file['redshift'])
        knnwidth = PointSigmaIQR(zb, truez)
        self.knnsigma = knnwidth.evaluate()
        knnbi = PointBias(zb, truez)
        self.knnbias = knnbi.evaluate()
        knnout = PointOutlierRate(zb, truez)
        self.knnoutrate = knnout.evaluate()
        print(f"sigIQR: {self.knnsigma}\nbias: {self.knnbias}\nout rate: {self.knnoutrate}")

        # break into bins based on zb and calculate SOM N(z) estimates
        # for each bin

        ## remove zb<0.3
        # lowzmask = (zb<0.3)
        # goodzb = zb[mask]
        ## break into N bins
        # binidxs = np.argsort(goodzb)
        # numgood = len(goodzb)
        # numperbin = numgood//Nbins
        # for i in range(Nbins):
        #    binmask = np.logical_and(bindixs > i * numperbin,
        #                             bindixs<=(i + 1) * numperbin)

    def make_test_file(self):
        """make test file for a small set of DC2 data,
        use the median depth from WFD pixels to set the
        magnitude errors
        """
        testfile = "/global/cfs/cdirs/lsst/groups/PZ/users/sschmidt/DDFSTUFF/three_hpix_9044_9301_10070_subset_for_wfd.pq"
        m5mask = np.zeros(len(self.coadd_depths), dtype=bool)
        for i, xm5 in enumerate(self.coadd_depths):
            if type(xm5) == list:
                m5mask[i] = 1
        tmpm5vals = self.coadd_depths[m5mask]
        wfd_m5vals = np.array(np.median(tmpm5vals), dtype=float)
        m5dict = {}
        for i, filt in enumerate(self.filternames):
            m5dict[f"{filt}"] = float(wfd_m5vals[i])
        #  read in the wfd truth data
        rawdata = pd.read_parquet(testfile)
        make_errs = pz_ddf_errors(rawdata, m5dict, 71)
        df = make_errs.run()

        #  replace non-detections
        for band, lim in zip(self.bands, wfd_m5vals):
            onesig = lim + 1.747425  # one sigma is five sigma + 1.747425
            # mask = np.isinf(trainfile['u'])
            df.loc[np.isinf(df[f'{band}'])] = 99.0
            df.loc[np.isinf(df[f'{band}_err'])] = onesig
        return df

    def make_training_file(self):
        """make training file for vvds, cosmos, and deep2
        """
        # eyeballed values for the fields, fix later!
        # hardcoding stuff for now, generalize later!
        # NOTE: cosmos and vvds are deep fields, looks like DEEP2 is not!
        # cosmos_ra = 150.1
        # cosmos_dec = 2.18
        # deep2f3_ra = 352.5
        # deep2f3_dec = 0.0
        # vvds_f2_ra = 36.5
        # vvds_f2_dec = -4.5
        # infiledict = {'cosmos': "/global/cfs/cdirs/lsst/groups/PZ/users/sschmidt/DDFSTUFF/MOCK_COSMOS.pq",
        #            'deep2': "/global/cfs/cdirs/lsst/groups/PZ/users/sschmidt/DDFSTUFF/MOCK_DEEP2.pq",
        #            'vvds': "/global/cfs/cdirs/lsst/groups/PZ/users/sschmidt/DDFSTUFF/MOCK_VVDS.pq"
        #            }

        # cosmos_pix = pixel_from_radec(cosmos_ra, cosmos_dec, 64)
        # deep2_pix = pixel_from_radec(deep2f3_ra, deep2f3_dec, 64)
        # vvds_pix = pixel_from_radec(vvds_f2_ra, vvds_f2_dec, 64)
        # pixdict = {'cosmos': cosmos_pix, 'deep2': deep2_pix, 'vvds': vvds_pix}
        # radict = {'cosmos': cosmos_ra, 'deep2': deep2f3_ra, 'vvds': vvds_f2_ra}
        # decdict = {'cosmos': cosmos_dec, 'deep2': deep2f3_dec, 'vvds': vvds_f2_dec}
        # pixeldict = {'cosmos': cosmos_pix, 'deep2': deep2_pix, 'vvds': vvds_pix}
        # surveys = ['cosmos', 'deep2', 'vvds']
        # surveys = ['cosmos', 'deep2']
        # surveym5dict = {}
        df = None
        for survey in self.surveylist:
            tmpra, tmpdec = self.radecdict[survey]
            survey_pixel = pixel_from_radec(tmpra, tmpdec, 64)
            m5vals = np.array(self.coadd_depths[survey_pixel], dtype=float)  # needed because photerr was complaining about float64!
            m5dict = {}
            for i, filt in enumerate(self.filternames):
                m5dict[f"{filt}"] = float(m5vals[i])
            # surveym5dict[f'{survey}'] = m5dict
            rawdata = pd.read_parquet(self.filedict[survey])
            make_errs = pz_ddf_errors(rawdata, m5dict, 9192)
            if df is None:
                df = make_errs.run()
            else:
                tmpdf = make_errs.run()
                df = pd.concat([df, tmpdf])

        #  replace non-detections
        for band, lim in zip(self.bands, m5vals):
            onesig = lim + 1.747425  # one sigma is five sigma + 1.747425
            # mask = np.isinf(trainfile['u'])
            df.loc[np.isinf(df[f'{band}'])] = 99.0
            df.loc[np.isinf(df[f'{band}_err'])] = onesig
        return df


class pz_ddf_errors(object):
    """quick class to make photometric errors for the DDF datasets
    """

    def __init__(self, data, m5dict, seed=87, colnamesdict=None):
        """
        Parameters:
        data: pandas dataframe
          input data with bands for errors to be added
        m5dict: dict
          dictionary of 5 sigma limiting mags for the bands in the LSST error model
        seed: int
          random seed
        """
        self.data = data
        self.m5dict = m5dict
        self.seed = seed
        if colnamesdict is not None:
            self.coldict = colnamesdict

    def run(self):
        """
        Returns:
        outdf: pandas dataframe
          dataframe with scattered errors and photom

            """
        numvis = dict(u=1, g=1, r=1, i=1, z=1, y=1)

        errmod = photerr.LsstErrorModel(m5=self.m5dict, nVisYr=numvis, nYrObs=1)
        newmags = errmod(self.data, random_state=self.seed)
        return newmags
