"""Render registered MANO and GS S1 surfaces, including an isolated thumb, in MuJoCo."""
import argparse
import csv
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation
from gs_s1_mujoco import hand_scene_xml
from gs_s1_rviz import get_package_share_directory

HERE = Path(__file__).resolve().parent


def normals(v, faces):
    face = np.cross(v[faces[:,1]] - v[faces[:,0]], v[faces[:,2]] - v[faces[:,0]])
    n = np.zeros_like(v)
    for col in faces.T:
        np.add.at(n, col, face)
    return n / np.maximum(np.linalg.norm(n,axis=1,keepdims=True),1e-12)


def add_mesh(root, label, start, faces):
    ids, remap = np.unique(faces, return_inverse=True)
    tri = remap.reshape(-1,3)
    v = start[ids]
    element = ET.SubElement(root.find("asset"),"mesh",name=label)
    element.set("vertex"," ".join(map(str,v.ravel())))
    element.set("normal"," ".join(map(str,normals(v,tri).ravel())))
    element.set("face"," ".join(map(str,tri.ravel())))
    return ids, tri


def main(surface, out, preview=None, compare_tracks=None):
    source = np.load(surface)
    track = np.load(HERE / "trajectories.npz")
    np.testing.assert_array_equal(source["frame"],track["frame"])
    # Same palm frame as the benchmark. Only rotate and translate; no hand-specific scale.
    pts = source["joints"] - source["joints"][:,:1]
    h = track["raw_human_palm_m"]
    u, _, vt = np.linalg.svd(pts.transpose(0,2,1) @ h)
    orient = u @ vt
    np.testing.assert_allclose(np.linalg.det(orient), 1., atol=1e-6)
    np.testing.assert_allclose(pts @ orient,h,atol=1e-6)
    vertices = (source["vertices"]-source["joints"][:,:1]) @ orient
    assets = Path.home() / "Projects/tendon-spin/assets"
    mesh_dir = Path(get_package_share_directory("dexhand_description")) / "meshes/gs_s1"
    root = hand_scene_xml(assets,mesh_dir,"right")
    palm = root.find(".//body[@name='palm']")
    world = root.find("worldbody")
    world.remove(world.find("geom[@name='floor_collision']"))
    human_meshes = {}
    for name,faces,group in (("mano_full",source["faces"],0),("mano_thumb",source["thumb_faces"],1)):
        human_meshes[name]=add_mesh(root,name,vertices[0],faces)
        ET.SubElement(palm,"geom",name=name,type="mesh",mesh=name,mass="0",contype="0",conaffinity="0",
                      rgba=".12 .72 .90 1",group=str(group))
    model=mj.MjModel.from_xml_string(ET.tostring(root,encoding="unicode"))
    for g in range(model.ngeom):
        if model.geom(g).name in human_meshes:
            continue
        body=model.body(model.geom_bodyid[g]).name
        model.geom_group[g] = 2 if body.startswith("th") else 3 if body=="palm" or body[:2] in ("ff","mf","rf","lf") else 5
        model.geom_rgba[g]=[.96,.46,.12,1] if body.startswith("th") else [.8,.63,.4,1]
    model.vis.global_.offwidth=432
    model.vis.global_.offheight=350
    model.vis.quality.offsamples=2
    model.vis.headlight.ambient[:]=.50
    data=mj.MjData(model)
    mj.mj_forward(model,data)
    palm_id=model.body("palm").id
    origin=data.xpos[palm_id].copy()
    R=data.xmat[palm_id].reshape(3,3).copy()
    renderer=mj.Renderer(model,width=432,height=350)
    def camera(target,distance,azimuth,elevation):
        cam=mj.MjvCamera()
        cam.lookat[:]=origin+R@target
        cam.distance,cam.azimuth,cam.elevation=distance,azimuth,elevation
        return cam
    main_cam=camera(np.array([-.008,.028,.088]),.33,-26,-21)
    thumb_cam=camera(np.array([-.037,.049,.063]),.192,-4,-16)
    def option(groups):
        opt=mj.MjvOption()
        opt.geomgroup[:]=0
        opt.geomgroup[groups]=1
        opt.sitegroup[:]=0
        return opt
    hand,robot,thuman,trobot=map(option,([0],[2,3],[1],[2]))
    def render(cam,opt):
        renderer.update_scene(data,camera=cam,scene_option=opt)
        renderer.scene.flags[mj.mjtRndFlag.mjRND_SHADOW]=0
        return renderer.render().copy()
    def mix(a,b):
        # Equal exposure prevents surfaces closer to the camera hiding the other hand.
        return np.rint(.5*a+.5*b).astype(np.uint8)
    def update_model(i):
        for name,(vids,tri) in human_meshes.items():
            mesh=model.mesh(name).id
            vadr=model.mesh_vertadr[mesh]
            nadr=model.mesh_normaladr[mesh]
            assert model.mesh_vertnum[mesh]==model.mesh_normalnum[mesh]==len(vids)
            rotation=Rotation.from_quat(np.roll(model.mesh_quat[mesh],-1)).as_matrix()
            local=(vertices[i,vids]-model.mesh_pos[mesh])@rotation
            if i==0:
                np.testing.assert_allclose(local,model.mesh_vert[vadr:vadr+len(vids)],atol=1e-6)
            model.mesh_vert[vadr:vadr+len(vids)]=local
            model.mesh_normal[nadr:nadr+len(vids)]=normals(local,tri)
            mj.mjr_uploadMesh(model,renderer._mjr_context,mesh)
    def robot_pose(angles):
        for n,a in zip(track["joint_names"],angles):
            data.qpos[model.joint(n).qposadr.item()]=a
        for digit in ("FF","MF","RF","LF"):
            data.qpos[model.joint(digit+"J1").qposadr.item()]=data.qpos[model.joint(digit+"J2").qposadr.item()]
        mj.mj_forward(model,data)
    with (HERE/"thumb.csv").open() as f:
        angles=[row for row in csv.DictReader(f) if row["method"]=="dexpilot"]
    font_path="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    large=ImageFont.truetype(font_path,25,index=2)
    small=ImageFont.truetype(font_path,19,index=2)
    base=Image.new("RGB",(1320,466),(22,30,37))
    draw=ImageDraw.Draw(base)
    draw.text((9,4),"DexMV 芥末瓶  /  固定腕部的手型对齐",font=large,fill="white")
    for x, text in [(8,"人手  ·  MANO"),(448,"全手叠加  ·  MANO + GS S1"),(888,"仅拇指叠加  ·  放大侧视")]:
        draw.text((x,51),text,font=small,fill="#deeff4")
    draw.text((8,424),"青色为人手，橙色为 DexPilot 在 GS S1 上的关节目标。",font=small,fill="#cfdae4")
    def frame(i):
        update_model(i)
        robot_pose(track["dexpilot_active_qpos_rad"][i])
        a,b=render(main_cam,hand),render(main_cam,robot)
        c,d=render(thumb_cam,thuman),render(thumb_cam,trobot)
        img=base.copy()
        for x,arr in [(8,a),(448,mix(a,b)),(888,mix(c,d))]:
            img.paste(Image.fromarray(arr),(x,78))
        ui=ImageDraw.Draw(img)
        ui.text((1020,7),f"源帧 {track['frame'][i]}    {i/25:.2f} s",font=large,fill="white")
        row=angles[i]
        ui.text((845,426),f"末节弯曲   人手 {float(row['human_thumb_ip_bend_deg']):.0f}°    GS S1 {float(row['robot_thumb_distal_bend_deg']):.0f}°",font=small,fill="white")
        return img
    if compare_tracks:
        methods = ("dexpilot", "no_anchor", "weak_anchor")
        with np.load(compare_tracks) as comparison:
            np.testing.assert_array_equal(comparison["joint_names"],track["joint_names"])
            np.testing.assert_array_equal(comparison["frame"],track["frame"])
            controls = [comparison[m+"_active_qpos_rad"] for m in methods]
        base = Image.new("RGB",(888,824),(22,30,37))
        labels = ("MANO 人手","DexPilot 基线","指形与指间向量 · 无掌部定位","指形与指间向量 · 弱掌部定位")
        panels = [(8,44),(448,44),(8,424),(448,424)]
        ui=ImageDraw.Draw(base)
        ui.text((8,5),"DexMV 芥末瓶  /  右手 GS S1 V3.2",font=large,fill="white")
        for (x,y),label in zip(panels,labels):
            ui.text((x,y),label,font=small,fill="#deeff4")
        def compare_frame(i):
            update_model(i)
            images = [render(main_cam,hand)]
            for poses in controls:
                robot_pose(poses[i])
                images.append(render(main_cam,robot))
            img = base.copy()
            for (x,y),arr in zip(panels,images):
                img.paste(Image.fromarray(arr),(x,y+27))
            ImageDraw.Draw(img).text((645,6),f"源帧 {track['frame'][i]}   {i/25:.2f} s",font=large,fill="white")
            return img
        frame = compare_frame
    try:
        if preview:
            frame(0)
            frame(77).save(preview)
        else:
            proc=subprocess.Popen(["ffmpeg","-v","error","-y","-f","rawvideo","-pixel_format","rgb24",
                "-video_size",f"{base.width}x{base.height}","-framerate","25","-i","pipe:0","-c:v","libx264","-threads","3",
                "-pix_fmt","yuv420p","-crf","17","-movflags","+faststart",str(out)],stdin=subprocess.PIPE)
            try:
                for i in range(len(vertices)):
                    proc.stdin.write(frame(i).tobytes())
            finally:
                proc.stdin.close()
                code=proc.wait()
            if code: raise RuntimeError(f"ffmpeg returned {code}")
    finally:
        renderer.close()


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("surface",type=Path)
    parser.add_argument("out",type=Path)
    parser.add_argument("--preview",type=Path)
    parser.add_argument("--compare-tracks",type=Path,help="Render MANO and the three offline anchor experiment trajectories.")
    main(**vars(parser.parse_args()))
