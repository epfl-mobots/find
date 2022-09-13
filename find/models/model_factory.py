import find.models.tf_models as tfm

model_choices = {
    'PLSTM': tfm.PLSTM,
    'PLSTM_SHALLOW': tfm.PLSTM_SHALLOW,
    'PLSTM_2L': tfm.PLSTM_2L,
    'PLSTM_MULT_PREDS': tfm.PLSTM_MULT_PREDS,
    'PFW': tfm.PFW,
    'LCONV': tfm.LCONV,
    'LSTM': tfm.LSTM,
    'PLSTM_builder': tfm.PLSTM_model_builder,
    'PFW_builder': tfm.PFW_model_builder,
    'EncDec': tfm.EncDec,
}

backend = {
    'PLSTM': 'keras',
    'PLSTM_SHALLOW': 'keras',
    'PLSTM_2L': 'keras',
    'PLSTM_MULT_PREDS': 'keras',
    'PFW': 'keras',
    'LCONV': 'keras',
    'LSTM': 'keras',
    'PLSTM_builder': 'keras',
    'PFW_builder': 'keras',
    'EncDec': 'keras',
}


def available_models():
    return list(model_choices.keys())


class ModelFactory:
    def __call__(self, model_choice, input_shape, output_shape, args):
        return model_choices[model_choice](input_shape, output_shape, args)

    def model_backend(self, model_choice):
        return backend[model_choice]
