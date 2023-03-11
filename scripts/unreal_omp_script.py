import unreal

if __name__ == "__main__":
    imp_pct = 90
    size_ratio = 100
    imp_res_file = f"/home/chad/datasets/ThreeDeeOffice/imp_res/{imp_pct}percent_markers/run0/current_placement.txt"
    case_type = f"Pct{imp_pct}Marker"
    prefix = f"tag{case_type}"
    holder_prefix = f"holder{case_type}"
    print(f"Marker prefix: {prefix}")
    # marker poses from IMP
    f_stream = open(imp_res_file, "r")
    imp_data = f_stream.readlines()
    tag_num = len(imp_data) - 1
    print(f"Marker number {tag_num}")
    # set this to the height of the PlayerStart in UE4Editor
    # the height was 160 for the HM3D scenes we used.
    height = 175
    # the scales below adjust the size of tags being added to the scene
    # the decal_scale was [.01, .1651, .1275] for the HM3D scenes we used.    
    decal_scale = [.01, 0.13208, 0.102]
    # the cube_scale was [.01, .88, .7] for the HM3D scenes we used.    
    cube_scale = [.01, .67, .52]
    # set x and y cooridnates of the PlayerStart in UE4Editor
    # these were 0 for the HM3D scenes we used.
    player_x, player_y = -272.0, 64.999985
    for i in  range(1, 3):
        decal_scale[i] = decal_scale[i]*size_ratio/100
        cube_scale[i] = cube_scale[i]*size_ratio/100

    cube_mesh = unreal.EditorAssetLibrary.find_asset_data('/Engine/BasicShapes/Cube').get_asset()
    for i in range(tag_num):
        line = imp_data[i+1].strip()
        print(line)
        line = line.split()
        # scale 100 converting meters to centimeters
        x, y, theta = float(line[1])*100+player_x, -float(line[2])*100+player_y, -float(line[3])*180/3.14
        # get texture file
        if i<9:
            tex_path = f"/Game/Flying/tags/tag-00{i+1}_mat"
        else:
            tex_path = f"/Game/Flying/tags/tag-0{i+1}_mat"
        if unreal.EditorAssetLibrary.does_asset_exist(tex_path):
            my_tex = unreal.EditorAssetLibrary.find_asset_data(tex_path).get_asset()
            my_decal = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.DecalActor,[0,0,0])
            my_decal.set_actor_label(f"xxx{prefix}{i}")
            my_decal.set_decal_material(my_tex)
            my_decal.set_actor_location_and_rotation([x, y, height], unreal.Rotator(-90, 180, theta), False, True)
            my_decal.set_actor_scale3d(decal_scale)
            my_decal.set_folder_path(f"/{case_type}")
            my_decal.set_actor_label(f"{prefix}{i}")

            my_cube = unreal.EditorLevelLibrary.spawn_actor_from_object(cube_mesh, [0, 0, 0])
            my_cube.set_actor_label(f"xxx{holder_prefix}{i}")
            my_cube.set_actor_location_and_rotation([x, y, height], unreal.Rotator(-90, 180, theta), False, True)
            my_cube.set_actor_scale3d(cube_scale)
            my_cube.set_folder_path(f"/{case_type}")
            my_cube.set_actor_label(f"{holder_prefix}{i}")
            my_cube.set_mobility(unreal.ComponentMobility.MOVABLE)
        else:
            print(f"cannot find tex {tex_path}")
    f_stream.close()
