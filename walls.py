
CUSTOM = ((
				(  0.0, 180,     0), # 0
				(  0.0,   0,     0),
				( 67.5, 180, 103.5),
				( 67.5,  45,     0),
				(135.0, 180,     0),
				(157.5, 180,     0), # 5
				(261.0, 180, 103.5),
				(261.0,  45,     0),
				(450.0, 180, 103.5),
				(450.0,  45,     0),
				(450.0,   0,     0), # 10
				(553.5,  45, 103.5),
				(553.5,   0, 103.5),
				(450.0, 180, 225.0),
				(553.5,  45, 225.0),
				(553.5,   0, 225.0), # 15
		), (
				(0, 1, 10, 9, 7, 5, 4, 3),
				(0, 3, 2),
				(3, 4, 2),
				(5, 7, 6),
				(6, 7, 9, 8),
				(8, 9, 11),
				(9, 10, 12, 11),
				(8, 11, 14, 13),
				(11, 12, 15, 14),
		))

CUSTOM2 = ((
				(  0,    0,     0), # 0
				(555,    0,     0),
				(555,  150,     0),
				(480,  150,     0),
				(480, 8.75,     0),
				(360, 8.75,     0), # 5
				(240, 8.75,     0),
				(240,  240,     0),
				(  0,  240,     0),
				(240,  240, 75.14),
				(360,  240, 75.14), # 10
				(360,  240, 95.79),
				(480,  240, 95.79),
				(480,  150, 58.51),
				(480,  240,110.48),
				(555,  150, 58.51), # 15
				(555,  240,110.48),
				(555,    0,110.48),
		), (
				(0, 1, 2, 3, 4, 6, 7, 8), # 0
				(7, 6, 9),
				(9, 6, 5, 10),
				(10, 5, 11),
				(11, 5, 4, 12),
				(12, 13, 14), # 5
				(13, 4, 3),
				(3, 2, 15, 13),
				(14, 13, 15, 16),
				(16, 15, 2, 1, 17),
		))

CUSTOM2_PROB_1 = (
		CUSTOM2, ((
						(0, 40, 10),
						(0, 80, 40),
						(0, 40, 70),
						(0, 80, 100),
						(0, 40, 130),
						(0, 80, 160),
						(0, 40, 190),
						(0, 80, 220),
				),
				(2,),
				(7,),
		))

CUSTOM2_PROB_2 = (
		CUSTOM2, ((
						(2, 40, 10),
						(2, 80, 40),
						(2, 40, 70),
						(2, 80, 100),
						(2, 40, 130),
						(2, 80, 160),
						(2, 40, 190),
						(2, 80, 220),
				),
				(2,),
				(7,),
		))

CUSTOM2_PROB_3 = (
		CUSTOM2, ((
						(4, 40, 10),
						(4, 80, 40),
						(4, 40, 70),
						(4, 80, 100),
						(4, 40, 130),
						(4, 80, 160),
						(4, 40, 190),
						(4, 80, 220),
				),
				(2,),
				(7,),
		))
