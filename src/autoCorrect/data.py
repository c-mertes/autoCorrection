from autoCorrect import * # <----------------------------------------------change

def data():
        dr=data_utils.DataReader()
        counts = dr.read_gtex_kremer_merged()
        cook = data_utils.DataCooker(counts)
        data = cook.data("OutInjectionFC")
        return data
