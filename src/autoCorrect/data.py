from autoCorrect import * # <----------------------------------------------change

def data():
        dr=data_utils.DataReader()
        counts = dr.read_gtex_skin()
        cook = data_utils.DataCookerTrainValidTest(counts)
        data = cook.data()
        return data
