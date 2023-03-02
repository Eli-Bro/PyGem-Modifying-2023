﻿NDSummary.OnToolTipsLoaded("File:lib/Shape_Analysis.py",{234:"<div class=\"NDToolTip TInformation LPython\"><div class=\"TTSummary\">The following functions were originally extracted from Generate_INP.py, in which all pertained to shape analysis control points of different parts. The common code between these sections in the Generate_INP.py file were moved here and broken down into functions and helper functions to streamline the codebase.</div></div>",231:"<div class=\"NDToolTip TFunction LPython\"><div id=\"NDPrototype231\" class=\"NDPrototype WideForm\"><div class=\"PSection PParameterSection CStyle\"><table><tr><td class=\"PBeforeParameters\"><span class=\"SHKeyword\">def</span> generate_data_points(</td><td class=\"PParametersParentCell\"><table class=\"PParameters\"><tr><td class=\"PName first last\">part,</td></tr><tr><td class=\"PName first last\">PCA_1,</td></tr><tr><td class=\"PName first last\">PCA_2,</td></tr><tr><td class=\"PName first last\">filename,</td></tr><tr><td class=\"PName first last\">n</td></tr></table></td><td class=\"PAfterParameters\">)</td></tr></table></div></div><div class=\"TTSummary\">Helper function for the levator_shape_analysis and ICM_shape_analysis functions, and acts as the common point between the 2 sections</div></div>",232:"<div class=\"NDToolTip TFunction LPython\"><div id=\"NDPrototype232\" class=\"NDPrototype WideForm\"><div class=\"PSection PParameterSection CStyle\"><table><tr><td class=\"PBeforeParameters\"><span class=\"SHKeyword\">def</span> levator_shape_analysis(</td><td class=\"PParametersParentCell\"><table class=\"PParameters\"><tr><td class=\"PName first last\">PCA_1,</td></tr><tr><td class=\"PName first last\">PCA_2</td></tr></table></td><td class=\"PAfterParameters\">)</td></tr></table></div></div><div class=\"TTSummary\">This function takes in the PCA scores given in the Generate_INP.py file</div></div>",233:"<div class=\"NDToolTip TFunction LPython\"><div id=\"NDPrototype233\" class=\"NDPrototype WideForm\"><div class=\"PSection PParameterSection CStyle\"><table><tr><td class=\"PBeforeParameters\"><span class=\"SHKeyword\">def</span> ICM_shape_analysis(</td><td class=\"PParametersParentCell\"><table class=\"PParameters\"><tr><td class=\"PName first last\">PCA_1,</td></tr><tr><td class=\"PName first last\">PCA_2,</td></tr><tr><td class=\"PName first last\">ys,</td></tr><tr><td class=\"PName first last\">zs</td></tr></table></td><td class=\"PAfterParameters\">)</td></tr></table></div></div><div class=\"TTSummary\">This function takes in the PCA scores from the Generate_INP.py file as well as the ys and zs from levator_shape_analysis</div></div>"});