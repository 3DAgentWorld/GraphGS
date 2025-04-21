import bpy
import bmesh

# Define the maximum coordinates
max_x = 7.39
max_y = 2.61
max_z = 0.67
# Get the current active object
obj = bpy.context.active_object

# Make sure the object is a mesh
if obj.type == 'MESH':
    # Get a BMesh from the object's mesh data
    bm = bmesh.new()
    bm.from_mesh(obj.data)

    # Find vertices that exceed the maximum coordinates and delete them
    for v in bm.verts:
        if v.co.x > max_x or v.co.y > max_y or v.co.z > max_z or v.co.x<-max_x or v.co.y<-max_y or v.co.z<-max_z:
            bm.verts.remove(v)

    # Write the BMesh back to the object's mesh data
    bm.to_mesh(obj.data)
    bm.free()