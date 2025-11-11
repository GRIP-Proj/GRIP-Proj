def get_config(type = "default"):
    '''
     - return the config dict
    '''
    if type == "default":
        config = dict()
        force_closure = dict()
        force_closure['quality_method'] = 'force_closure'
        force_closure['num_cone_faces'] = 8
        force_closure['soft_fingers'] = 1
        force_closure['quality_type'] = 'quasi_static'
        force_closure['all_contacts_required']= 1
        force_closure['check_approach'] = False
        force_closure['torque_scaling'] = 0.01
        force_closure['wrench_norm_thresh'] = 0.001
        force_closure['wrench_regularizer'] = 0.0000000001
        config['metrics'] = dict()
        config['metrics']['force_closure'] = force_closure
        return config
    
    elif type == "graspgen_wrench_resistance":
        """
            "quality_method":"wrench_resistance",
            "friction_coef":0.7,
            "num_cone_faces":16,
            "soft_fingers":False,
            "check_approach":True,
            "target_wrench":1,
            as step() in evaluate_graspgen_grasp.py
        """
        config = dict()
        wrench_resistance = dict()
        wrench_resistance['quality_method'] = 'wrench_resistance'
        wrench_resistance['num_cone_faces'] = 16
        wrench_resistance['soft_fingers'] = False
        wrench_resistance['quality_type'] = None
        wrench_resistance['all_contacts_required']= None
        wrench_resistance['check_approach'] = True
        wrench_resistance['torque_scaling'] = None
        wrench_resistance['wrench_norm_thresh'] = None
        wrench_resistance['wrench_regularizer'] = None
        config['metrics'] = dict()
        config['metrics']['wrench_resistance'] = wrench_resistance
