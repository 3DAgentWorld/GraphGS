import bpy
import bmesh

ply_object = bpy.context.selected_objects[0]
# to transform opencv coord to blender
if ply_object.type == 'MESH':
    # 进入编辑模式
    bpy.context.view_layer.objects.active = ply_object
    bpy.ops.object.mode_set(mode='EDIT')

    # 获取BMesh对象
    bm = bmesh.from_edit_mesh(ply_object.data)

    # 遍历所有顶点并修改Y和Z坐标
    for vert in bm.verts:
        vert.co.y = -vert.co.y
        vert.co.z = -vert.co.z

        # 更新网格
    bmesh.update_edit_mesh(ply_object.data)

        # 回到对象模式
    bpy.ops.object.mode_set(mode='OBJECT')
else:
    print("the target is not a MESH")
