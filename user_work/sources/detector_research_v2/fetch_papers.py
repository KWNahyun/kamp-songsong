from pathlib import Path
import urllib.request,concurrent.futures,subprocess,json
R=Path(__file__).parent
papers={
'SET':'https://openaccess.thecvf.com/content/CVPR2025/papers/Sun_SET_Spectral_Enhancement_for_Tiny_Object_Detection_CVPR_2025_paper.pdf',
'PGD':'https://openaccess.thecvf.com/content/CVPR2025/papers/Bian_Feature_Information_Driven_Position_Gaussian_Distribution_Estimation_for_Tiny_Object_CVPR_2025_paper.pdf',
'UGS':'https://openaccess.thecvf.com/content/ICCV2025/papers/Sun_Uncertainty-Aware_Gradient_Stabilization_for_Small_Object_Detection_ICCV_2025_paper.pdf',
'DMEFS':'https://openaccess.thecvf.com/content/ICCV2025/papers/Sharma_DM-EFS_Dynamically_Multiplexed_Expanded_Features_Set_Form_for_Robust_and_ICCV_2025_paper.pdf',
'DQDETR':'https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09775.pdf',
'SRTOD':'https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02515.pdf',
'TOLF':'https://arxiv.org/pdf/2601.00617',
'FreqFusion':'https://arxiv.org/pdf/2408.12879',
'BDNet':'https://openaccess.thecvf.com/content/CVPR2026/papers/Guan_BDNetBio-Inspired_Dual-Backbone_Small_Object_Detection_Network_CVPR_2026_paper.pdf',
'NSFPN':'https://openaccess.thecvf.com/content/CVPR2026/papers/Yuan_Seeing_Through_the_Noise_Improving_Infrared_Small_Target_Detection_and_CVPR_2026_paper.pdf',
'DyFCLT':'https://openaccess.thecvf.com/content/CVPR2026/papers/Li_DyFCLT_Dynamic_Frequency-Decoupled_Cross-Modal_Learning_Transformer_for_Multimodal_Tiny_Object_CVPR_2026_paper.pdf'}
def run(kv):
 k,u=kv
 try:
  p=R/(k+'.pdf')
  if not p.exists():p.write_bytes(urllib.request.urlopen(u,timeout=40).read())
  subprocess.run(['pdftotext','-layout',str(p),str(R/(k+'.txt'))],check=True)
  return k,'ok'
 except Exception as e:return k,str(e)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as e:print(list(e.map(run,papers.items())))
(R/'downloaded_papers.json').write_text(json.dumps(papers,indent=2))
