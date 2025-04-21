import bpy
import math


def set_camera_position(frame, radius, height, axis, direction, camera_object):
    angle = frame * (2 * math.pi / 250)  # 250帧一圈
    if direction == 'NEGATIVE':
        angle = -angle

    if axis == 'X':
        x = height
        y = radius * math.cos(angle)
        z = radius * math.sin(angle)
        camera_object.location = (x, y, z)
        # 保持相机下边框与Y-Z平面平行
        camera_object.rotation_euler = (math.radians(90), 0, angle)
    elif axis == 'Y':
        x = radius * math.cos(angle)
        y = height
        z = radius * math.sin(angle)
        camera_object.location = (x, y, z)
        # 保持相机下边框与X-Z平面平行
        camera_object.rotation_euler = (math.radians(90), angle, 0)
    else:  # 默认绕Z轴旋转
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        z = height
        camera_object.location = (x, y, z)
        camera_object.rotation_euler = (math.radians(90), 0, angle + math.radians(90))


def main(set_mode, radius, height, mp4_path):
    # 设置渲染引擎为工作台
    bpy.context.scene.render.engine = 'BLENDER_WORKBENCH'

    # 获取已有的相机对象
    camera_object = bpy.data.objects['Camera']

    axis = 'Z'  # 绕哪个轴旋转，可选 'X', 'Y', 'Z'
    direction = 'POSITIVE'  # 旋转方向，可选 'POSITIVE', 'NEGATIVE'

    # 设置相机位置和旋转

    # 设置相机对准原点
    bpy.context.view_layer.objects.active = camera_object
    bpy.ops.object.constraint_add(type='TRACK_TO')
    track_to_constraint = camera_object.constraints[-1]
    track_to_constraint.target = bpy.data.objects['Cube']  # 假设原点物体是默认的Cube
    track_to_constraint.track_axis = 'TRACK_NEGATIVE_Z'
    track_to_constraint.up_axis = 'UP_Y'

    # 设置渲染输出路径和格式
    bpy.context.scene.render.filepath = mp4_path
    bpy.context.scene.render.image_settings.file_format = 'FFMPEG'
    bpy.context.scene.render.ffmpeg.format = 'MPEG4'
    bpy.context.scene.render.ffmpeg.codec = 'H264'
    bpy.context.scene.render.ffmpeg.constant_rate_factor = 'HIGH'
    bpy.context.scene.render.ffmpeg.ffmpeg_preset = 'GOOD'

    # 设置帧数
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = 250

    # 为每一帧设置相机位置
    for frame in range(bpy.context.scene.frame_start, bpy.context.scene.frame_end + 1):
        bpy.context.scene.frame_set(frame)
        set_camera_position(frame, radius, height, axis, direction, camera_object)
        camera_object.keyframe_insert(data_path="location", frame=frame)
        camera_object.keyframe_insert(data_path="rotation_euler", frame=frame)

    # 渲染动画
    if not set_mode:
        bpy.ops.render.render(animation=True)


if __name__ == "__main__":
    set_mode = True
    radius = 7
    height = 4
    mp4_path = r'D:\NIPS2024\visAnything_test.mp4'
    main(set_mode, radius, height, mp4_path)
