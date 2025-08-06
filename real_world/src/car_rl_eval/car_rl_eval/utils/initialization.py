import importlib
import os.path
import sys


def create_alg(**kwargs):
    alg_name = kwargs["algorithm"]
    alg_file_name = alg_name.lower()
    try:
        module = importlib.import_module(alg_file_name)
    except NotImplementedError:
        raise NotImplementedError("This algorithm does not exist")

    if hasattr(module, alg_name):
        alg_cls = getattr(module, alg_name)
        alg = alg_cls(**kwargs)
    else:
        raise NotImplementedError("This algorithm is not properly defined")

    print("Create algorithm successfully!")
    return alg


def create_apprfunc(**kwargs):
    apprfunc_name = kwargs["apprfunc"]
    apprfunc_file_name = apprfunc_name.lower()
    try:
        file = importlib.import_module('networks.' + apprfunc_file_name)
    except NotImplementedError:
        raise NotImplementedError("This apprfunc does not exist")

    # name = kwargs['name'].upper()

    name = formatter(kwargs["name"])
    # print(name)
    # print(kwargs)

    if hasattr(file, name):  #
        apprfunc_cls = getattr(file, name)
        apprfunc = apprfunc_cls(**kwargs)
    else:
        raise NotImplementedError("This apprfunc is not properly defined")

    # print("--Initialize appr func: " + name + "...")
    return apprfunc


def create_buffer(**kwargs):
    buffer_file_name = kwargs["buffer_name"].lower()
    try:
        module = importlib.import_module("training." + buffer_file_name)
    except NotImplementedError:
        raise NotImplementedError("This buffer does not exist")

    buffer_name = formatter(buffer_file_name)

    if hasattr(module, buffer_name):  #
        buffer_cls = getattr(module, buffer_name)  #
        buffer = buffer_cls(**kwargs)
    else:
        raise NotImplementedError("This buffer is not properly defined")

    print("Create buffer successfully!")
    return buffer

def create_sampler(**kwargs):
    sampler_file_name = kwargs["sampler_name"].lower()
    try:
        module = importlib.import_module("training." + sampler_file_name)
    except NotImplementedError:
        raise NotImplementedError("This sampler does not exist")

    sampler_name = formatter(sampler_file_name)

    if hasattr(module, sampler_name):  #
        sampler_cls = getattr(module, sampler_name)  #
        sampler = sampler_cls(**kwargs)
    else:
        raise NotImplementedError("This sampler is not properly defined")

    print("Create sampler successfully!")
    return sampler


def create_evaluator(env,**kwargs):
    evaluator_file_name = kwargs["evaluator_name"].lower()
    try:
        module = importlib.import_module("training." + evaluator_file_name)
    except NotImplementedError:
        raise NotImplementedError("This evaluator does not exist")

    evaluator_name = formatter(evaluator_file_name)

    if hasattr(module, evaluator_name):  #
        evaluator_cls = getattr(module, evaluator_name)  #
        evaluator = evaluator_cls(env,**kwargs)
    else:
        raise NotImplementedError("This evaluator is not properly defined")

    print("Create evaluator successfully!")
    return evaluator


def create_trainer(alg, sampler,buffer, evaluator,**kwargs):
    trainer_file_name = kwargs["trainer"].lower()
    try:
        module = importlib.import_module("training." + trainer_file_name)
    except NotImplementedError:
        raise NotImplementedError("This trainer does not exist")

    trainer_name = formatter(trainer_file_name)

    if hasattr(module, trainer_name):  #
        trainer_cls = getattr(module, trainer_name)  #
        trainer = trainer_cls(alg, sampler,buffer, evaluator, **kwargs)
    else:
        raise NotImplementedError("This trainer is not properly defined")

    print("Create trainer successfully!")
    return trainer

def formatter(src: str, firstUpper: bool = True):
    arr = src.split("_")
    res = ""
    for i in arr:
        res = res + i[0].upper() + i[1:]

    if not firstUpper:
        res = res[0].lower() + res[1:]
    return res
