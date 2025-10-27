    def prepare_Y(self, y1_patterns=('W',)):
        """
        최종 Y 딕셔너리 병합 반환:
        {
          "Y0": { 'W': {'Gamma':(256,), 'Cx':(256,), 'Cy':(256,)}, 
                  'R': {'Gamma':(256,), 'Cx':(256,), 'Cy':(256,)},
                  'G': {'Gamma':(256,), 'Cx':(256,), 'Cy':(256,)},
                  'B': {'Gamma':(256,), 'Cx':(256,), 'Cy':(256,)},
                }
          "Y1": { 'W': (255,),
                  'R': (255,),
                  'G': (255,),
                  'B': (255,) 
                },
          "Y2": { 'Red': val, 
                  ..., 
                  'Western': val 
                }
        }
        """
        y0 = self.compute_Y0_struct()
        y1 = self.compute_Y1_struct(patterns=y1_patterns)
        y2 = self.compute_Y2_struct()
        
        return {"Y0": y0, "Y1": y1, "Y2": y2}


prepare_output.py를 위처럼 바꾸어서, VAC_dataset.py를 아래와 같이 하는 방향으로 하고자 합니다.


    def _collect(self):
        for pk in self.pk_list:
            x_builder = VACInputBuilder(pk)
            y_builder = VACOutputBuilder(pk)
            X = x_builder.prepare_X_delta()   # {"dLUT": {...}, "meta": {...}}
            Y = y_builder.prepare_Y(y1_patterns=('W',))    # {"Y0": {...}, "Y1": {...}, "Y2": {...}}
            
            self.samples.append({
                "pk": pk, 
                "X": X, 
                "Y": Y
            })

이 방향으로 하면 또 어느부분을 수정해야 할까요?
