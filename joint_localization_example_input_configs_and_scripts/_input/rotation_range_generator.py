def generate_rotation_ranges():
    # Symmetric ranges
    spine_01_y = 5
    spine_01_z = 5
    spine_02_y = 5
    spine_02_z = 5
    spine_03_y = 5
    spine_03_z = 5
    neck_y = 15
    neck_z = 15
    head_y = 15
    head_z = 15

    leg_lift = 10

    finger_rotations = [
        ((  -15,   +15), (  -10,   +10), (  -10,     +10)),
        ((   -5,   +15), (  -10,   +10), (  -2.5,   +2.5)),
        ((   -5,   +15), (  -10,   +10), (  -2.5,   +2.5)),
    ]

    # Rotation ranges
    rr = {
        # Root
        'Root' : ((+90-10, +90+10), (+0, +0), (+180, +180)), # ((  90,   90), (   0,    0), (+180, +180)) is the default orientation
        # Spine, neck and head
        'spine_01'    : ((  -5,   +5), (  -spine_01_y,   +spine_01_y), (  -spine_01_z,   +spine_01_z)),
        'spine_02'    : ((  -5,   +5), (  -spine_02_y,   +spine_02_y), (  -spine_02_z,   +spine_02_z)),
        'spine_03'    : ((  -5,   +5), (  -spine_03_y,   +spine_03_y), (  -spine_03_z,   +spine_03_z)),
        'neck_01'     : ((  -5,   +5), (  -    neck_y,   +    neck_y), (  -neck_z,   +neck_z)),
        'head'        : ((  -5,   +5), (  -    head_y,   +    head_y), (  -head_z,   +head_z)),
        # Legs
        'thigh'     : ((  -leg_lift, +leg_lift), ( -7.5,   +7.5), (  -leg_lift, +leg_lift)),
        'calf'      : ((  -leg_lift, +leg_lift), ( -7.5,   +7.5), (  -leg_lift/2, +leg_lift/2)),
        # Feet
        'foot'      : ((  -5,     +5), (  -5,    +5), (   -5,    +5)),
        'ball'      : ((  -5,     +5), (  -5,    +5), (   -5,    +5)),
        # Shoulders and arms
        'clavicle'  : ((  -7.5,       +7.5), (  -7.5,   +7.5), (  -2.5,   +2.5)),
        'upperarm'  : ((  -15/2,     +30/2), (  -45/2,     +45/2), (  -7.5/2,   +45/2)),
        'lowerarm'  : ((  -15/2,     +30/2), (  -45/2,     +45/2), (  -7.5/2,   +45/2)),
        'hand'      : ((  -15/2,     +30/2), (  -45/2,     +45/2), (  -7.5/2,   +45/2)),

        # Finger joints
        'thumb_01'  : finger_rotations[0],
        'thumb_02'  : finger_rotations[1],
        'thumb_03'  : finger_rotations[2],
        'index_01'  : finger_rotations[0],
        'index_02'  : finger_rotations[1],
        'index_03'  : finger_rotations[2],
        'middle_01' : finger_rotations[0],
        'middle_02' : finger_rotations[1],
        'middle_03' : finger_rotations[2],
        'ring_01'   : finger_rotations[0],
        'ring_02'   : finger_rotations[1],
        'ring_03'   : finger_rotations[2],
        'pinky_01'  : finger_rotations[0],
        'pinky_02'  : finger_rotations[1],
        'pinky_03'  : finger_rotations[2],
    }

    def symm_inv(rotrng):
        return (rotrng[0], (-rotrng[1][1], -rotrng[1][0]), (-rotrng[2][1], -rotrng[2][0]))

    # For each bone, for each axis, define a range of rotation.
    rr = {
      # 'pelvis'      : None, # NOTE: Pelvis rotation is not (0, 0, 0) in rest pose. It's fine to skip this.
        'Root'        : rr['Root'],
        # Spine, neck and head
        'spine_01'    : rr['spine_01'],
        'spine_02'    : rr['spine_02'],
        'spine_03'    : rr['spine_03'],
        'neck_01'     : rr['neck_01'],
        'head'        : rr['head'],
        # Legs
        'thigh_l'     : rr['thigh'],
        'thigh_r'     : symm_inv(rr['thigh']),
        'calf_l'      : rr['calf'],
        'calf_r'      : symm_inv(rr['calf']),
        # Feet
        'foot_l'      : rr['foot'],
        'foot_r'      : symm_inv(rr['foot']),
        'ball_l'      : rr['ball'],
        'ball_r'      : symm_inv(rr['ball']),
        # Shoulders and arms
        'clavicle_l'  : rr['clavicle'],
        'clavicle_r'  : symm_inv(rr['clavicle']),
        'upperarm_l'  : rr['upperarm'],
        'upperarm_r'  : symm_inv(rr['upperarm']),
        'lowerarm_l'  : rr['lowerarm'],
        'lowerarm_r'  : symm_inv(rr['lowerarm']),
        'hand_l'      : rr['hand'],
        'hand_r'      : symm_inv(rr['hand']),
        # Finger joints TODO fix finger right hand
        'thumb_01_l'  : rr['thumb_01'],
        'thumb_01_r'  : symm_inv(rr['thumb_01']),
        'thumb_02_l'  : rr['thumb_02'],
        'thumb_02_r'  : symm_inv(rr['thumb_02']),
        'thumb_03_l'  : rr['thumb_03'],
        'thumb_03_r'  : symm_inv(rr['thumb_03']),
        'index_01_l'  : rr['index_01'],
        'index_01_r'  : symm_inv(rr['index_01']),
        'index_02_l'  : rr['index_02'],
        'index_02_r'  : symm_inv(rr['index_02']),
        'index_03_l'  : rr['index_03'],
        'index_03_r'  : symm_inv(rr['index_03']),
        'middle_01_l' : rr['middle_01'],
        'middle_01_r' : symm_inv(rr['middle_01']),
        'middle_02_l' : rr['middle_02'],
        'middle_02_r' : symm_inv(rr['middle_02']),
        'middle_03_l' : rr['middle_03'],
        'middle_03_r' : symm_inv(rr['middle_03']),
        'ring_01_l'   : rr['ring_01'],
        'ring_01_r'   : symm_inv(rr['ring_01']),
        'ring_02_l'   : rr['ring_02'],
        'ring_02_r'   : symm_inv(rr['ring_02']),
        'ring_03_l'   : rr['ring_03'],
        'ring_03_r'   : symm_inv(rr['ring_03']),
        'pinky_01_l'  : rr['pinky_01'],
        'pinky_01_r'  : symm_inv(rr['pinky_01']),
        'pinky_02_l'  : rr['pinky_02'],
        'pinky_02_r'  : symm_inv(rr['pinky_02']),
        'pinky_03_l'  : rr['pinky_03'],
        'pinky_03_r'  : symm_inv(rr['pinky_03']),
    }

    return rr