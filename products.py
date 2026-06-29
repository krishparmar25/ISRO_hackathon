MOON_RADIUS_M = 1_737_400.0

MOON_SRS_WKT = """
GEOGCRS["Moon 2000",
  DATUM["Moon 2000",
    ELLIPSOID["Moon 2000 IAU IAG",1737400,0,
      LENGTHUNIT["metre",1]]],
  PRIMEM["Reference Meridian",0],
  CS[ellipsoidal,2],
    AXIS["geodetic latitude (Lat)",north],
    AXIS["geodetic longitude (Lon)",east],
    ANGLEUNIT["degree",0.0174532925199433]]
"""

MOON_PROJ4 = "+proj=longlat +R=1737400 +no_defs"

PDS4_DTYPE_MAP = {
    "IEEE754MSBSingle": ">f4",
    "IEEE754LSBSingle": "<f4",
    "IEEE754MSBDouble": ">f8",
    "IEEE754LSBDouble": "<f8",
    "MSBInteger": ">i2",
    "LSBInteger": "<i2",
    "UnsignedMSBInteger": ">u2",
    "UnsignedLSBInteger": "<u2",
    "SignedMSBInteger": ">i2",
    "SignedLSBInteger": "<i2",
    "MSBUnsignedInteger": ">u2",
    "LSBUnsignedInteger": "<u2",
    "ComplexMSB16": ">c8",
    "ComplexLSB16": "<c8",
}

BAND_ORDERINGS = {
    "re_h_im_h_re_v_im_v": ((0, 1), (2, 3)),
    "re_h_re_v_im_h_im_v": ((0, 2), (1, 3)),
    "re_h_im_v_im_h_re_v": ((0, 2), (3, 1)),
}

