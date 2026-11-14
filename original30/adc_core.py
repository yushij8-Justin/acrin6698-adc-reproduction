"""Frozen ADC computation A and strict public-DICOM/ROI contract. No outcome I/O."""
from __future__ import annotations
import numpy as np

B_VALUES=np.array([0.,100.,600.,800.])
class GateError(ValueError):
 def __init__(self,code,detail='',status='HOLD'):
  self.code=code;self.status=status;super().__init__(code+(': '+detail if detail else ''))
def require(condition,code,detail='',status='HOLD'):
 if not condition:raise GateError(code,detail,status)
def fit(signal,b_values=(0,100,600,800)):
 s=np.asarray(signal,dtype=np.float64);b=np.asarray(b_values,dtype=np.float64)
 require(s.ndim>=1 and s.shape[0]==4 and np.array_equal(b,B_VALUES),'B_VALUES')
 shape=s.shape[1:];x=s.reshape(4,-1);a=np.full(x.shape[1],np.nan);reason=np.zeros(x.shape[1],np.uint8)
 finite=np.isfinite(x).all(axis=0);nonpositive=finite&(x<=0).any(axis=0);valid=finite&~nonpositive
 reason[~finite]=3;reason[nonpositive]=1;a[nonpositive]=0
 bc=b-b.mean()
 with np.errstate(all='ignore'):a[valid]=-(bc[:,None]*np.log(x[:,valid])).sum(axis=0)/(bc@bc)
 bad=valid&~np.isfinite(a);reason[bad]=3;a[bad]=np.nan
 zero=valid&~bad&(a<=1e-12);reason[zero]=2;a[zero]=0
 return {'adc':a.reshape(shape),'reason':reason.reshape(shape),'high':(a>.01).reshape(shape)}
def identity(d,expected):
 for keyword,key in [('PatientID','patient_id'),('StudyInstanceUID','study_uid'),('SeriesInstanceUID','series_uid'),('ClinicalTrialTimePointID','visit')]:
  require(hasattr(d,keyword) and str(getattr(d,keyword))==str(expected[key]),'IDENTITY_'+keyword,status='STOP')
 require(hasattr(d,'FrameOfReferenceUID'),'MISSING_FRAME_OF_REFERENCE',status='STOP')
def scaling(d,adc=False):
 require(not any(el.tag in (0x00409096,0x00283000) for el in d.iterall()),'EXTRA_RWVM_OR_LUT' if adc else 'TRACE_ALTERNATIVE_MAPPING_REQUIRES_REVIEW')
 has_rs=hasattr(d,'RescaleSlope');has_ri=hasattr(d,'RescaleIntercept')
 if not adc and not has_rs and not has_ri:
  require(str(d.SOPClassUID)=='1.2.840.10008.5.1.4.1.1.4','UNSUPPORTED_IDENTITY_SIGNAL_ENCODING')
  return 1.,0.,1.  # No public transform: retain stored classic-MR signal, not physical calibration.
 require(has_rs and has_ri,'MISSING_PUBLIC_SCALING')
 rs=float(d.RescaleSlope);ri=float(d.RescaleIntercept)
 require(np.isfinite(rs) and rs>0 and np.isfinite(ri),'INVALID_PUBLIC_SCALING')
 factor=1.
 if adc:
  unit=str(getattr(d,'RescaleType','')).replace(' ','').replace('²','2').replace('^2','2')
  units={'10^-3mm2/s':.001,'mm2/s':1.}
  require(unit in units,'UNKNOWN_ADC_UNIT',unit);factor=units[unit]
 return rs,ri,factor

def _frames(dslist,expected):
 require(len(dslist)>0,'NO_DICOM')
 ids=[str(d.SOPInstanceUID) for d in dslist];require(len(ids)==len(set(ids)),'DUPLICATE_SOP')
 frames=[]
 for d in dslist:
  identity(d,expected);pix=d.pixel_array
  if hasattr(d,'PerFrameFunctionalGroupsSequence'):
   shared=d.SharedFunctionalGroupsSequence[0]
   if pix.ndim==2:pix=pix[None]
   require(pix.ndim==3 and len(pix)==len(d.PerFrameFunctionalGroupsSequence),'MULTIFRAME_COUNT')
   for j,f in enumerate(d.PerFrameFunctionalGroupsSequence):
    def fg(name):
     seq=getattr(f,name,getattr(shared,name,None));require(seq is not None and len(seq)>0,'MISSING_GEOMETRY',name);return seq[0]
    if hasattr(f,'SegmentIdentificationSequence'):require(int(f.SegmentIdentificationSequence[0].ReferencedSegmentNumber)==1,'MULTIPLE_SEGMENTS')
    pm=fg('PixelMeasuresSequence');frames.append({'pixels':pix[j],'position':np.asarray(fg('PlanePositionSequence').ImagePositionPatient,float),'orientation':np.asarray(fg('PlaneOrientationSequence').ImageOrientationPatient,float),'spacing':np.asarray(pm.PixelSpacing,float),'thickness':float(pm.SliceThickness),'dataset':d})
  else:
   require(pix.ndim==2,'CLASSIC_MR_DIMENSIONS')
   for attr in ['ImagePositionPatient','ImageOrientationPatient','PixelSpacing','SliceThickness']:require(hasattr(d,attr),'MISSING_GEOMETRY',attr)
   frames.append({'pixels':pix,'position':np.asarray(d.ImagePositionPatient,float),'orientation':np.asarray(d.ImageOrientationPatient,float),'spacing':np.asarray(d.PixelSpacing,float),'thickness':float(d.SliceThickness),'dataset':d})
 return frames

def load_volume(dslist,expected,kind='ADC'):
 frames=_frames(dslist,expected);first=frames[0];ori=first['orientation'];spacing=first['spacing']
 require(ori.shape==(6,) and spacing.shape==(2,) and np.isfinite(ori).all() and np.isfinite(spacing).all() and (spacing>0).all(),'INVALID_GEOMETRY')
 normal=np.cross(ori[:3],ori[3:]);require(abs(np.linalg.norm(ori[:3])-1)<=1e-5 and abs(np.linalg.norm(ori[3:])-1)<=1e-5 and abs(np.linalg.norm(normal)-1)<=1e-5 and abs(ori[:3]@ori[3:])<=1e-5,'INVALID_ORIENTATION')
 for f in frames:
  require(f['pixels'].shape==first['pixels'].shape,'MATRIX_MISMATCH')
  require(np.max(np.abs(f['orientation']-ori))<=1e-5,'ORIENTATION_MISMATCH')
  require(np.max(np.abs(f['spacing']-spacing))<=1e-5,'SPACING_MISMATCH')
  require(np.isfinite(f['position']).all(),'NONFINITE_GEOMETRY')
  require(str(f['dataset'].FrameOfReferenceUID)==str(first['dataset'].FrameOfReferenceUID),'FRAME_OF_REFERENCE_MISMATCH',status='STOP')
 frames.sort(key=lambda f:float(f['position']@normal));positions=np.array([f['position'] for f in frames]);z=positions@normal
 require(len(z)==len(np.unique(np.round(z,5))),'DUPLICATE_SLICE')
 if len(z)>1:
  dz=float(np.median(np.diff(z)));require(dz>0,'NONPOSITIVE_SLICE_SPACING')
  if kind=='ADC':require(np.max(np.abs(np.diff(z)-dz))<=.01,'NONREGULAR_SLICE_SPACING')
  residual=(positions-positions[0])-(z-z[0])[:,None]*normal
  require(np.max(np.linalg.norm(residual,axis=1))<=.01,'SLICE_INPLANE_DRIFT')
 else:dz=first['thickness']
 pixels=np.stack([f['pixels'] for f in frames]);physical=pixels.astype(np.float64)
 if kind=='ADC':
  scales=[scaling(f['dataset'],True) for f in frames];require(len(set(scales))==1,'SLICE_SCALING_VARIES')
  for i,(rs,ri,k) in enumerate(scales):physical[i]=(pixels[i].astype(np.float64)*rs+ri)*k
 else:
  scales=[]
  if kind=='MASK':require(set(np.unique(pixels))<={0,1},'NONBINARY_MASK')
 metadata={'kind':kind,'shape':list(pixels.shape),'scales':scales,'slice_spacing':dz,'frame_of_reference':str(first['dataset'].FrameOfReferenceUID),'rescale_type':str(getattr(first['dataset'],'RescaleType',''))}
 return {'pixels':pixels,'physical':physical,'positions':positions,'orientation':ori,'spacing':spacing,'normal':normal,'datasets':dslist,'frames':frames,'for_uid':str(first['dataset'].FrameOfReferenceUID),'metadata':metadata}

def align(reference,moving,full=False):
 require(reference['pixels'].shape[1:]==moving['pixels'].shape[1:],'MATRIX_MISMATCH')
 require(reference['for_uid']==moving['for_uid'],'FRAME_OF_REFERENCE_MISMATCH',status='STOP')
 require(np.max(np.abs(reference['orientation']-moving['orientation']))<=1e-5,'ORIENTATION_MISMATCH')
 require(np.max(np.abs(reference['spacing']-moving['spacing']))<=1e-5,'SPACING_MISMATCH')
 dist=np.linalg.norm(moving['positions'][:,None,:]-reference['positions'][None,:,:],axis=-1)
 ix=np.argmin(dist,axis=1);require(np.max(dist[np.arange(len(ix)),ix])<=.01,'POSITION_MISMATCH')
 require(len(np.unique(ix))==len(ix),'DUPLICATE_ALIGNED_SLICE')
 if full:require(len(ix)==len(reference['positions']),'MISSING_SLICE')
 out=np.zeros_like(reference['pixels'],dtype=moving['pixels'].dtype);out[ix]=moving['pixels'];return out

def trace_signal(dslist,reference,expected):
 require(len(dslist)>0,'NO_DICOM')
 sop=[str(d.SOPInstanceUID) for d in dslist];require(len(sop)==len(set(sop)),'DUPLICATE_SOP')
 require(all(hasattr(d,'DiffusionBValue') for d in dslist),'MISSING_B_VALUE')
 bs=sorted({float(d.DiffusionBValue) for d in dslist});require(np.array_equal(bs,B_VALUES),'B_VALUES')
 signal=np.zeros((4,*reference['pixels'].shape),float);raw=np.zeros_like(signal);seen=set();scale_entries=[]
 for d in dslist:
  identity(d,expected)
  for attr in ['ImageOrientationPatient','ImagePositionPatient','PixelSpacing']:require(hasattr(d,attr),'MISSING_GEOMETRY',attr)
  require(str(d.FrameOfReferenceUID)==reference['for_uid'],'FRAME_OF_REFERENCE_MISMATCH',status='STOP')
  require(np.max(np.abs(np.asarray(d.ImageOrientationPatient,float)-reference['orientation']))<=1e-5,'ORIENTATION_MISMATCH')
  require(np.max(np.abs(np.asarray(d.PixelSpacing,float)-reference['spacing']))<=1e-5,'SPACING_MISMATCH')
  dist=np.linalg.norm(reference['positions']-np.asarray(d.ImagePositionPatient,float),axis=1);zi=int(dist.argmin());require(dist[zi]<=.01,'POSITION_MISMATCH')
  bi=bs.index(float(d.DiffusionBValue));require((bi,zi) not in seen,'DUPLICATE_B_SLICE');seen.add((bi,zi))
  pix=d.pixel_array;require(pix.shape==reference['pixels'].shape[1:],'MATRIX_MISMATCH')
  rs,ri,_=scaling(d);raw[bi,zi]=pix;signal[bi,zi]=pix.astype(float)*rs+ri;scale_entries.append((rs,ri))
 require(len(seen)==4*len(reference['positions']),'MISSING_B_SLICE')
 return {'signal':signal,'raw':raw,'b_values':bs,'positions':reference['positions'].copy(),'scales':sorted(set(scale_entries)),'all_public_rescale_explicit':all(hasattr(d,'RescaleSlope') and hasattr(d,'RescaleIntercept') for d in dslist),'identity_no_public_transform_instances':sum(not hasattr(d,'RescaleSlope') and not hasattr(d,'RescaleIntercept') for d in dslist)}

def roi_readout(public_adc,reference_adc,mask):
 p=np.asarray(public_adc,float);u=np.asarray(reference_adc,float);m=np.asarray(mask,bool)
 require(p.shape==u.shape==m.shape,'ROI_SHAPE')
 require(m.any(),'EMPTY_ROI');pv=p[m];uv=u[m]
 require(np.isfinite(pv).all() and np.isfinite(uv).all(),'NONFINITE_ROI')
 require((pv>=0).all(),'NEGATIVE_PUBLIC_ADC');require((uv>=0).all(),'NEGATIVE_REFERENCE_ADC')
 require(np.any(pv>0),'ALL_ZERO_PUBLIC_ADC');require(np.any(uv>0),'ALL_ZERO_REFERENCE_ADC')
 common=(pv>0)&(uv>0);pm=float(pv.mean());um=float(uv.mean())
 return {'P_mean':pm,'U_mean':um,'roi_voxels':int(m.sum()),'difference_output_units':1000*(pm-um),'public_zero':int((pv==0).sum()),'reference_zero':int((uv==0).sum()),'zero_support_mismatch':int(np.count_nonzero((pv==0)!=(uv==0))),'common_positive_n':int(common.sum()),'common_positive_P_mean':float(pv[common].mean()) if common.any() else None,'common_positive_U_mean':float(uv[common].mean()) if common.any() else None}

def audit_references(dslist,known_sop_uids):
 known=set(known_sop_uids);image=set();study=set()
 def recurse(d):
  if hasattr(d,'ReferencedSOPInstanceUID'):
   uid=str(d.ReferencedSOPInstanceUID);cl=str(getattr(d,'ReferencedSOPClassUID',''))
   if cl=='1.2.840.10008.3.1.2.3.1':study.add(uid)
   else:image.add(uid)
  for el in d:
   if el.VR=='SQ':
    for item in el.value:recurse(item)
 for d in dslist:recurse(d)
 unresolved=sorted(image-known)
 require(not unresolved,'UNRESOLVED_IMAGE_SOURCE',','.join(unresolved[:3]))
 return {'image_reference_count':len(image),'study_reference_count':len(study),'unresolved':unresolved}

def threshold_audit(b0_raw,public_zero,mask,slope):
 b=np.asarray(b0_raw,float);z=np.asarray(public_zero,bool);m=np.asarray(mask,bool)
 require(b.shape==z.shape==m.shape,'THRESHOLD_SHAPE')
 return {name:{'threshold':threshold,'predicted_zero_voxels':int(((b<=threshold)&m).sum()),'zero_mismatch_voxels':int((((b<=threshold)!=z)&m).sum())} for name,threshold in [('raw10',10.),('raw10_times_source_RS',10.*float(slope))]}
