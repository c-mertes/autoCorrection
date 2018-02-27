from autoCorrect import * # <----------------------------------------------change

def data():
        dr=data_utils.DataReader()
        #counts = dr.read_gtex_kremer_merged()
        counts = dr.read_gtex_skin()
        cook = data_utils.DataCooker(counts, inject_on_pred=True)
        data = cook.data("OutInjectionFC")
        return data
