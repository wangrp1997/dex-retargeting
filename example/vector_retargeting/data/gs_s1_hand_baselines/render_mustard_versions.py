"""Compare annotated hand/bottle poses and the two free-bottle physics replays."""
import argparse
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import xml.etree.ElementTree as ET

import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation

from gs_s1_dexmv import FPS, load_sequence, make_scene
from gs_s1_rviz import get_package_share_directory
from render_mano_overlay import add_mesh, normals


def render(args):
    args.hand = 'right'
    tracks = [dict(np.load(args.results/(v+'.npz'))) for v in ('v3.2','v3.3')]
    ref = tracks[0]
    for key in ('frame','joint_names','human_palm','reference_xyz','reference_rotation','wrist_xyz','wrist_rotation'):
        np.testing.assert_array_equal(ref[key],tracks[1][key])
    mano=np.load(args.surface)
    ids=ref['frame']
    np.testing.assert_array_equal(ids,mano['frame'])
    pts=mano['joints']-mano['joints'][:,:1]
    u,_,vt=np.linalg.svd(pts.transpose(0,2,1)@ref['human_palm'])
    np.testing.assert_allclose(pts@(u@vt),ref['human_palm'],atol=1e-6)
    vertices=(mano['vertices']-mano['joints'][:,:1])@(u@vt)
    meshes=Path(get_package_share_directory('dexhand_description'))/'meshes/gs_s1'
    original=make_scene(args,meshes,False,load_sequence(args.dexmv_root)[4])
    with TemporaryDirectory() as temp:
        xml=Path(temp)/'scene.xml'
        mj.mj_saveLastXML(str(xml),original)
        root=ET.parse(xml).getroot()
    vid,faces=add_mesh(root,'mano',vertices[0],mano['faces'])
    palm=root.find(".//body[@name='palm']")
    ET.SubElement(palm,'geom',name='mano',type='mesh',mesh='mano',group='2',rgba='.12 .73 .90 1',
                  contype='0',conaffinity='0',mass='0')
    asset=root.find('asset')
    skin=args.dexmv_root/'hand_imitation/env/models/assets/ycb/visual/006_mustard_bottle'
    ET.SubElement(asset,'mesh',name='skin',file=str(skin/'textured_simple_blender.msh'))
    ET.SubElement(asset,'texture',name='skin_tex',type='2d',file=str(skin/'texture_map.png'))
    ET.SubElement(asset,'material',name='bottle_mat',texture='skin_tex',specular='.12')
    obj=root.find(".//body[@name='mustard']")
    for geom in obj.findall('geom'):
        geom.set('group','5')
    ET.SubElement(obj,'geom',type='mesh',mesh='skin',material='bottle_mat',group='0',contype='0',conaffinity='0',mass='0')
    model=mj.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
    np.testing.assert_array_equal(model.jnt_qposadr,original.jnt_qposadr)
    bottle=model.body('mustard').id
    for g,bid in enumerate(model.geom_bodyid):
        name=model.body(bid).name or ''
        if bid in (0,bottle) or model.geom(g).name=='mano':
            continue
        model.geom_group[g]=1 if name=='palm' or name[:2] in ('th','ff','mf','rf','lf') else 5
        model.geom_rgba[g]=[.96,.47,.13,1] if name.startswith('th') else [.84,.73,.56,1]
    data=mj.MjData(model)
    W,H=420,290
    model.vis.global_.offwidth=W
    model.vis.global_.offheight=H
    model.vis.quality.offsamples=2
    model.vis.headlight.ambient[:]=.50
    robot_opt,human_opt=mj.MjvOption(),mj.MjvOption()
    for opt,groups in ((robot_opt,[0,1]),(human_opt,[0,2])):
        opt.geomgroup[:]=0
        opt.geomgroup[groups]=1
        opt.sitegroup[:]=0
    mesh=model.mesh('mano').id
    va,na=model.mesh_vertadr[mesh],model.mesh_normaladr[mesh]
    rot=Rotation.from_quat(np.roll(model.mesh_quat[mesh],-1)).as_matrix()
    adr=[model.joint(n).qposadr.item() for n in ref['joint_names']]
    passive=[model.joint(f+'J1').qposadr.item() for f in ('FF','MF','RF','LF')]
    parent=[model.joint(f+'J2').qposadr.item() for f in ('FF','MF','RF','LF')]
    obj_adr=model.joint('bottle_free').qposadr.item()
    qs=np.roll(Rotation.from_matrix(ref['reference_rotation']).as_quat(),1,axis=1)
    joints_world=ref['human_palm']@ref['wrist_rotation'].transpose(0,2,1)+ref['wrist_xyz'][:,None]
    hand_center=joints_world.mean(axis=1)
    center=gaussian_filter1d(.55*hand_center+.45*ref['reference_xyz'],3,axis=0)
    dist=gaussian_filter1d(.42+.95*np.linalg.norm(hand_center-ref['reference_xyz'],axis=1),3)
    all_pos=np.stack([hand_center,ref['reference_xyz'],tracks[0]['object_xyz'],tracks[1]['object_xyz']])
    wide_center=.5*(all_pos.min(axis=0)+all_pos.max(axis=0))
    wide_dist=.53+1.65*np.linalg.norm(all_pos-wide_center,axis=2).max(axis=0)
    renderer=mj.Renderer(model,width=W,height=H)
    def camera(target,dist):
        cam=mj.MjvCamera()
        cam.lookat[:]=target
        cam.azimuth,cam.elevation,cam.distance=16,-27,dist
        return cam
    def image(cam,opt):
        renderer.update_scene(data,camera=cam,scene_option=opt)
        renderer.scene.flags[mj.mjtRndFlag.mjRND_REFLECTION]=0
        return Image.fromarray(renderer.render())
    font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    title=ImageFont.truetype(font,23,index=2)
    small=ImageFont.truetype(font,18,index=2)
    base=Image.new('RGB',(1284,734),(22,30,37))
    draw=ImageDraw.Draw(base)
    draw.text((8,4),'DexMV 芥末瓶搬运  /  博雅 GS S1 右手',font=title)
    for j,label in enumerate(('DexMV 人手 · MANO 重建','GS S1 · v3.2','GS S1 · v3.3')):
        draw.text((6+426*j,42),label,font=small,fill='#d5edf3')
    draw.text((6,362),'人手与标注瓶位 · 广角',font=small)
    for j in (1,2):
        draw.text((6+426*j,362),'自由瓶 · 实际物理回放',font=small)
    draw.text((425,9),'上排用演示瓶位比较手型，下排看瓶子是否被抓起',font=small,fill='#b9cdd2')
    def frame(i):
        local=(vertices[i,vid]-model.mesh_pos[mesh])@rot
        if i==0:
            np.testing.assert_allclose(local,model.mesh_vert[va:va+len(vid)],atol=1e-6)
        model.mesh_vert[va:va+len(vid)]=local
        model.mesh_normal[na:na+len(vid)]=normals(local,faces)
        mj.mjr_uploadMesh(model,renderer._mjr_context,mesh)
        near=camera(center[i],dist[i])
        wide=camera(wide_center[i],wide_dist[i])
        out=base.copy()
        ui=ImageDraw.Draw(out)
        ui.text((1008,8),f'源帧 {ids[i]}    {i/FPS:.2f} s',font=small)
        for j,tr in enumerate((ref,*tracks)):
            x=6+426*j
            data.qpos[:]=tr['qpos'][i]
            if j:
                mj.mj_forward(model,data)
                np.testing.assert_allclose(data.xpos[bottle],tr['object_xyz'][i],atol=1e-7)
                out.paste(image(wide,robot_opt),(x,390))
                value=tr['clearance_m'][i]*1000
                msg='手瓶接触' if tr['hand_contact'][i] else '无受力接触'
                data.qpos[adr]=tr['active_qpos_rad'][i]
                data.qpos[passive]=data.qpos[parent]
            else:
                value=ref['reference_clearance_m'][i]*1000
                msg='标注轨迹'
            data.qpos[obj_adr:obj_adr+3]=ref['reference_xyz'][i]
            data.qpos[obj_adr+3:obj_adr+7]=qs[i]
            mj.mj_forward(model,data)
            np.testing.assert_allclose(data.xmat[model.body('palm').id].reshape(3,3),ref['wrist_rotation'][i],atol=1e-6)
            np.testing.assert_allclose(data.xpos[model.body('palm').id],ref['wrist_xyz'][i],atol=1e-6)
            out.paste(image(near,robot_opt if j else human_opt),(x,68))
            if not j:
                out.paste(image(wide,human_opt),(x,390))
            ui.text((x,692),f'瓶底相对桌面  {value:.1f} mm   ·   {msg}',font=small,fill='#cae2e4')
        return out
    try:
        if args.preview is not None:
            frame(list(ids).index(args.preview)).save(args.output)
            return
        with subprocess.Popen(['ffmpeg','-nostdin','-v','error','-y','-f','rawvideo','-pixel_format','rgb24',
                   '-video_size','1284x734','-framerate',str(FPS),'-i','pipe:0','-an','-c:v','libx264',
                   '-threads','3','-pix_fmt','yuv420p','-crf','17','-movflags','+faststart',str(args.output)],
                   stdin=subprocess.PIPE) as proc:
            for i in range(len(ids)):
                proc.stdin.write(frame(i).tobytes())
            proc.stdin.close()
            if proc.wait():
                raise RuntimeError('ffmpeg encoding failed')
    finally:
        renderer.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('surface','results','output'):
        p.add_argument(name,type=Path)
    p.add_argument('--preview',type=int)
    p.add_argument('--dexmv-root',type=Path,default=Path.home()/'Projects/dexmv-sim')
    p.add_argument('--assets',type=Path,default=Path.home()/'Projects/tendon-spin/assets')
    render(p.parse_args())
