# pzddf

A few thrown together classes and functions to generate some 5 sigma depth limits for VVDS, DEEP2, and COSMOS,
plus some methods to use `photerr` to add uncertainties to those datasets and a fiducial WFD dataset.  We can
then train a SOM and a knnpdf informer and calculate redshifts.  Once we sort the redshifts into tomographic bins
we can develop a metric (probably just difference in mean redshift between estimated and true tomo bin) and use
that to test sensitivity to the DDF and WFD depths.


