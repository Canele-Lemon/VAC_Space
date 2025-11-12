12bit의 R_Low, G_Low, B_Low값이 저장되어있는 
LOW_LUT_CSV   = r"D:\00 업무\00 가상화기술\00 색시야각 보상 최적화\VAC algorithm\Gen VAC\Random VAC\4. 기준 LUT + OFFSET\기준 LUT\LUT_low_values_SIN1300.csv"
를 인터폴레이션 한 R_High, G_High, B_High 열과 붙여 GrayLevel_window	R_Low	R_High	G_Low	G_High	B_Low	B_High 제목행으로 시작하는 csv 파일이 되게 하고 싶습니다.

그리고 최종적으로 

    def write_default_data(file, table_format):
    if table_format == "txt":
        default_data = """{    																			
0,	1,	1,																	
//DRV_valc_ctrl_t																			
{																			
	//DRV_valc_pattern_ctrl_t																		
	{																		
		11,	1,																
		{																	
			{1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	}, //line 0
			{0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	}, //line 1
			{1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	}, //line 2
			{0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	}, //line 3
			{1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	}, //line 4
			{0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	}, //line 5
			{1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	}, //line 6
			{0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	}, //line 7
		}																	
	},																		
	// DRV_valc_sat_ctrl_t																		
	{																		
		{0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	},	
	},																		
	// DRV_valc_hpf_ctrl_t																		
	{																		
		{0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	},	
		1,																	
	},																		
},																			
"""
    
    elif table_format == "json":
        default_data = """{																					
"DRV_valc_major_ctrl"	:	[	0,	1	],																
"DRV_valc_pattern_ctrl_0"	:	[	5,	1	],																
"DRV_valc_pattern_ctrl_1"	:	[	[	1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	0	],	
			[	0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	1	],	
			[	1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	0	],	
			[	0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	1	],	
			[	1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	0	],	
			[	0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	1	],	
			[	1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	0	],	
			[	0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	1	]      	 ],
"DRV_valc_sat_ctrl"	:	[	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0	],		
"DRV_valc_hpf_ctrl_0"	:	[	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0	],		
"DRV_valc_hpf_ctrl_1"	:		1,																		
"""
    file.write(default_data)
    
    
    
def write_LUT_data(file, input_file_path, table_format):
    LUT = pd.read_csv(input_file_path)
    
    if table_format == "txt":        
        channels = {
            "R Channel": ['R_Low', 'R_High'],
            "G Channel": ['G_Low', 'G_High'],
            "B Channel": ['B_Low', 'B_High']
        }
        
        file.write("//LUT\n{\n")

        for channel_name, cols in channels.items():
            file.write(f"\t// {channel_name}\n\t")
            file.write("{\n")
            for col in cols:
                file.write("\t\t{\n")
                data = LUT[col].values
                reshaped_data = np.reshape(data, (256, 16))
                for row in reshaped_data:
                    formatted_row = '\t\t' + ',\t'.join(map(lambda x: str(int(x)), row)) + ','
                    file.write(formatted_row + '\n')
                file.write("\t\t},\n")
            file.write("\t},\n")
        file.write("},\n};")
        
    elif table_format == "json":
        channels = {
            "RchannelLow": 'R_Low',
            "RchannelHigh": 'R_High',
            "GchannelLow": 'G_Low',
            "GchannelHigh": 'G_High',
            "BchannelLow": 'B_Low',
            "BchannelHigh": 'B_High'
        }
        
        for i, (channel_name, col) in enumerate(channels.items()):
            file.write(f'"{channel_name}"\t:\t[\t')
            data = LUT[col].values
            reshaped_data = np.reshape(data, (256, 16)).tolist()

            for row_index, row in enumerate(reshaped_data):
                formatted_row = ',\t'.join(map(lambda x: str(int(x)), row))
                if row_index == 0:
                    file.write(f'{formatted_row},\n')
                elif row_index == len(reshaped_data) - 1:
                    file.write(f'\t\t\t{formatted_row}')
                else:
                    file.write(f'\t\t\t{formatted_row},\n')

            if i == len(channels) - 1:
                file.write("\t]\n")
            else:
                file.write("\t],\n")

        file.write("}")

def main():
    Tk().withdraw()
    
    table_format= input("Select Table Format (txt/json): ").strip().lower()
    if table_format not in ["txt", "json"]:
        print("@INFO: Invalid table format selected. Exiting.")
        return
    
    input_file_path = askopenfilename(title="Select Input CSV File", filetypes=[("CSV Files", "*.csv")])
    if not input_file_path:
        print("@INFO: Input file not selected. Exiting.")
        return
    
    output_file_path = f"../Gen DGA/STEP4 LUT Formatting & Loading/LUT_DGA.{table_format}"
    
    with open(output_file_path, 'w') as f:
        if table_format == "txt":
            write_default_data(f, "txt")
            write_LUT_data(f, input_file_path, "txt")
        elif table_format == "json":
            write_default_data(f, "json")
            write_LUT_data(f, input_file_path, "json")

    print(f"@INFO: Data has been successfully written to {output_file_path}")
    absolute_path = os.path.abspath(output_file_path)
    
    # Windows 환경에서 파일 열기
    try:
        os.startfile(absolute_path)  
    except FileNotFoundError:
        print(f'@ERROR: File not found')

if __name__ == "__main__":
    main()

이 함수를 사용해 table_format == "json"으로 포멧변환하여 json 파일이 생성되도록 해 주세요. 
