# A Cython wrapper for vlfeat
# Author: Sameh Khamis

from __future__ import division
from libcpp.string cimport string
import numpy as np
cimport numpy as np

np.import_array()


cdef extern from 'host.h':
    int VL_FALSE
    int VL_TRUE
    ctypedef int vl_bool
    ctypedef long long unsigned vl_size

cdef extern from 'hog.h':
    ctypedef enum VlHogVariant:
        VlHogVariantDalalTriggs
        VlHogVariantUoctti
    ctypedef struct VlHog:
        pass
    VlHog* vl_hog_new(VlHogVariant, vl_size, vl_bool)
    void vl_hog_set_use_bilinear_orientation_assignments(VlHog*, vl_bool)
    void vl_hog_put_image(VlHog*, const float*, vl_size, vl_size, vl_size, vl_size)
    void vl_hog_put_polar_field(VlHog*, const float*, const float*, vl_bool, vl_size, vl_size, vl_size)
    vl_size vl_hog_get_glyph_size(const VlHog*)
    vl_size vl_hog_get_width(VlHog*)
    vl_size vl_hog_get_height(VlHog*)
    vl_size vl_hog_get_dimension(const VlHog*)
    void vl_hog_render(const VlHog*, float*, const float*, vl_size, vl_size)
    void vl_hog_extract(VlHog*, float*)
    void vl_hog_delete(VlHog*)

cdef extern from 'dsift.h':
    ctypedef struct VlDsiftFilter:
        pass
    ctypedef struct VlDsiftDescriptorGeometry:
        int numBinT
        int numBinX
        int numBinY
        int binSizeX
        int binSizeY
    ctypedef struct VlDsiftKeypoint:
        double x
        double y
        double s
        double norm
    VlDsiftFilter* vl_dsift_new (int, int)
    void vl_dsift_set_steps(VlDsiftFilter*, int, int)
    void vl_dsift_set_geometry (VlDsiftFilter*, const VlDsiftDescriptorGeometry*)
    void vl_dsift_set_bounds(VlDsiftFilter*, int, int, int, int)
    void vl_dsift_set_flat_window(VlDsiftFilter*, vl_bool)
    void vl_dsift_set_window_size(VlDsiftFilter*, double)
    void vl_dsift_process(VlDsiftFilter*, const float*)
    int vl_dsift_get_keypoint_num(const VlDsiftFilter*)
    int vl_dsift_get_descriptor_size(const VlDsiftFilter*)
    const VlDsiftKeypoint* vl_dsift_get_keypoints(const VlDsiftFilter*)
    const float* vl_dsift_get_descriptors(const VlDsiftFilter*)
    void vl_dsift_transpose_descriptor(float*, const float*, int, int, int)
    void vl_dsift_delete(VlDsiftFilter*)

cdef extern from 'sift.h':
    ctypedef struct VlSiftFilt:
        pass
    ctypedef struct VlSiftKeypoint:
        int o
        float x
        float y
        float sigma
    VlSiftFilt* vl_sift_new(int, int, int, int, int)
    int vl_sift_process_first_octave(VlSiftFilt*, const float*)
    int vl_sift_process_next_octave(VlSiftFilt*)
    void vl_sift_detect(VlSiftFilt*)
    void vl_sift_set_peak_thresh(VlSiftFilt*, double)
    void vl_sift_set_edge_thresh(VlSiftFilt*, double)
    void vl_sift_set_norm_thresh(VlSiftFilt*, double)
    void vl_sift_set_magnif(VlSiftFilt*, double)
    void vl_sift_set_window_size(VlSiftFilt*, double)
    const VlSiftKeypoint* vl_sift_get_keypoints (VlSiftFilt*)
    int vl_sift_get_nkeypoints(const VlSiftFilt*)
    int vl_sift_get_octave_index(const VlSiftFilt*)
    void  vl_sift_keypoint_init(const VlSiftFilt*, VlSiftKeypoint*, double, double, double)
    int vl_sift_calc_keypoint_orientations(VlSiftFilt*, double[4], const VlSiftKeypoint*)
    void vl_sift_calc_keypoint_descriptor(VlSiftFilt*, float*, const VlSiftKeypoint*, double)
    void vl_sift_delete(VlSiftFilt*)


def hog(np.ndarray[np.float32_t, ndim = 3] im, int cellsize, variant = 'UoCTTI', int num_orientations = 9, bilinear_orientations = False):
    cdef VlHogVariant cvariant = VlHogVariantDalalTriggs if variant == 'DalalTriggs' else VlHogVariantUoctti
    cdef VlHog* hog = vl_hog_new(cvariant, num_orientations, VL_TRUE)
    vl_hog_set_use_bilinear_orientation_assignments(hog, VL_TRUE if bilinear_orientations else VL_FALSE)
    cdef np.ndarray[np.float32_t, ndim = 3] imf = im.T.copy()
    vl_hog_put_image(hog, <float*> imf.data, imf.shape[2], imf.shape[1], imf.shape[0], cellsize)
    
    cdef vl_size hog_height = vl_hog_get_height(hog)
    cdef vl_size hog_width = vl_hog_get_width(hog)
    cdef vl_size hog_dimension = vl_hog_get_dimension(hog)
    cdef np.ndarray[np.float32_t, ndim = 3] feat = np.zeros([hog_dimension, hog_height, hog_width], dtype=np.float32)
    vl_hog_extract(hog, <float*> feat.data)
    
    vl_hog_delete(hog)
    return feat.T.copy()


def hog_polar_field(np.ndarray[np.float32_t, ndim = 3] im, int cellsize, variant = 'UoCTTI', int num_orientations = 9, directed = False, bilinear_orientations = False):
    cdef VlHogVariant cvariant = VlHogVariantDalalTriggs if variant == 'DalalTriggs' else VlHogVariantUoctti
    cdef VlHog* hog = vl_hog_new(cvariant, num_orientations, VL_TRUE)
    vl_hog_set_use_bilinear_orientation_assignments(hog, VL_TRUE if bilinear_orientations else VL_FALSE)
    cdef np.ndarray[np.float32_t, ndim = 2] modulus = im[:, :, 0].T.copy()
    cdef np.ndarray[np.float32_t, ndim = 2] angle = im[:, :, 1].T.copy()
    vl_hog_put_polar_field(hog, <float*> modulus.data, <float*> angle.data, VL_TRUE if directed else VL_FALSE, modulus.shape[1], modulus.shape[0], cellsize)
    
    cdef vl_size hog_height = vl_hog_get_height(hog)
    cdef vl_size hog_width = vl_hog_get_width(hog)
    cdef vl_size hog_dimension = vl_hog_get_dimension(hog)
    cdef np.ndarray[np.float32_t, ndim = 3] feat = np.zeros([hog_dimension, hog_height, hog_width], dtype=np.float32)
    vl_hog_extract(hog, <float*> feat.data)
    
    vl_hog_delete(hog)
    return feat.T.copy()


def hog_render(np.ndarray[np.float32_t, ndim = 3] feat, variant = 'UoCTTI', int num_orientations = 9):
    cdef VlHogVariant cvariant = VlHogVariantDalalTriggs if variant == 'DalalTriggs' else VlHogVariantUoctti
    cdef VlHog* hog = vl_hog_new(cvariant, num_orientations, VL_TRUE)
    cdef vl_size glyph_size = vl_hog_get_glyph_size(hog)
    
    cdef np.ndarray[np.float32_t, ndim = 3] featf = feat.T.copy()
    cdef np.ndarray[np.float32_t, ndim = 2] hogim = np.zeros([featf.shape[1] * glyph_size, featf.shape[2] * glyph_size], dtype=np.float32)
    vl_hog_render(hog, <float*> hogim.data, <float*> featf.data, featf.shape[2], featf.shape[1])
    
    vl_hog_delete(hog)
    return hogim.T.copy()


def dsift(np.ndarray[np.float32_t, ndim=2] im, step = 1, size = 3, bounds = [], norm = False, fast = False, window_size = -1, float_descriptors = False, geometry = [4, 4, 8]):
    cdef np.ndarray[np.float32_t, ndim = 2] imf = im.T.copy()
    cdef VlDsiftFilter* dsift = vl_dsift_new(imf.shape[1], imf.shape[0])
    cdef VlDsiftDescriptorGeometry geom
    geom.binSizeY, geom.binSizeX = size, size if np.size(size) == 1 else size[:2]
    geom.numBinY, geom.numBinX, geom.numBinT = geometry
    vl_dsift_set_geometry(dsift, &geom)
    if np.size(step) != 2: step = [step, step]
    vl_dsift_set_steps(dsift, step[1], step[0])
    if np.size(bounds) == 4: vl_dsift_set_bounds(dsift, max(bounds[1], 0), max(bounds[0], 0), min(bounds[3], imf.shape[1] - 1), min(bounds[2], imf.shape[0] - 1))
    vl_dsift_set_flat_window(dsift, VL_TRUE if fast else VL_FALSE)
    if window_size >= 0: vl_dsift_set_window_size(dsift, window_size)
    cdef int nframes = vl_dsift_get_keypoint_num(dsift)
    cdef int sdescrs = vl_dsift_get_descriptor_size(dsift)
    
    vl_dsift_process(dsift, <float*> imf.data)
    cdef const VlDsiftKeypoint* cframes = vl_dsift_get_keypoints(dsift)
    cdef const float* cdescrs = vl_dsift_get_descriptors(dsift)
    
    cdef np.ndarray[np.float32_t, ndim = 2] frames = np.zeros([nframes, 3 if norm else 2], dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim = 2] descrs = np.zeros([nframes, sdescrs], dtype=np.float32)
    cdef int k
    for k in range(nframes):
        frames[k, :] = [cframes[k].y, cframes[k].x, cframes[k].norm] if norm else [cframes[k].y, cframes[k].x]
        vl_dsift_transpose_descriptor(<float*> descrs.data + k * sdescrs, cdescrs + k * sdescrs, geom.numBinT, geom.numBinX, geom.numBinY)
    
    vl_dsift_delete(dsift)
    descrs = np.minimum(512 * descrs, 255)
    return (frames, descrs) if float_descriptors else (frames, descrs.astype(np.uint8))


def sift(np.ndarray[np.float32_t, ndim=2] im, octaves = -1, levels = 3, first_octave = 0, peak_thresh = -1, edge_thresh = -1, norm_thresh = -1, magnif = -1, window_size = -1, inframes = None, force_orientations = False, float_descriptors = False):
    cdef np.ndarray[np.float32_t, ndim = 2] imf = im.T.copy()
    cdef VlSiftFilt* filt = vl_sift_new(imf.shape[1], imf.shape[0], octaves, levels, first_octave)
    if peak_thresh >= 0: vl_sift_set_peak_thresh(filt, peak_thresh)
    if edge_thresh >= 0: vl_sift_set_edge_thresh(filt, edge_thresh)
    if norm_thresh >= 0: vl_sift_set_norm_thresh(filt, norm_thresh)
    if magnif >= 0: vl_sift_set_magnif(filt, magnif)
    if window_size >= 0: vl_sift_set_window_size(filt, window_size)
    
    cdef int k, q, nangles, nkeys
    cdef const VlSiftKeypoint* keys
    cdef double[4] angles
    cdef const VlSiftKeypoint* kp
    cdef VlSiftKeypoint* ikp
    cdef float[128] buf
    cdef np.ndarray[np.float32_t, ndim = 1] rbuf = np.empty([128], dtype=np.float32)
    cdef int fkeys = 1 if np.ndim(inframes) == 2 and np.shape(inframes)[0] == 4 else 0
    frameslist = []
    descrslist = []
    
    cdef int err = vl_sift_process_first_octave(filt, <float*> imf.data)
    while not err:
        if fkeys:
            nkeys = np.shape(inframes)[1]
        else:
            vl_sift_detect(filt)
            keys = vl_sift_get_keypoints(filt)
            nkeys = vl_sift_get_nkeypoints(filt)
        
        for k in range(nkeys):
            if fkeys:
                vl_sift_keypoint_init(filt, ikp, inframes[1, k], inframes[0, k], inframes[2, k])
                if vl_sift_get_octave_index(filt) != k.o: break
                kp = ikp
                if force_orientations:
                    nangles = vl_sift_calc_keypoint_orientations(filt, angles, kp)
                else:
                    angles[0] = np.pi / 2 - inframes[3, k]
                    nangles = 1
            else:
                kp = keys + k
                nangles = vl_sift_calc_keypoint_orientations(filt, angles, kp)
            
            for q in range(nangles):
                vl_sift_calc_keypoint_descriptor(filt, buf, kp, angles[q])
                vl_dsift_transpose_descriptor(<float*> rbuf.data, buf, 8, 4, 4)
                frameslist.append([kp.y, kp.x, kp.sigma, np.pi / 2 - angles[q]])
                descrslist.append(rbuf.copy())
        
        err = vl_sift_process_next_octave(filt)
    
    vl_sift_delete(filt)
    cdef np.ndarray[np.float32_t, ndim = 2] frames = np.array(frameslist, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim = 2] descrs = np.array(descrslist, dtype=np.float32)
    descrs = np.minimum(512 * descrs, 255)
    return (frames, descrs) if float_descriptors else (frames, descrs.astype(np.uint8))


def lbp():
    pass


def mser():
    pass


def fisher():
    pass


def gmm():
    pass


def covdet():
    pass

