import misc.constants as cs
import os
import xml.etree.ElementTree as ET
import pandas as pd


def load_ohio(dataset, subject):
    """
    Load OhioT1DM training_old and testing files into a dataframe
    :param dataset: name of dataset
    :param subject: name of subject
    :return: dataframe
    """
    train_path, test_path = _compute_file_names(dataset, subject)

    [train_xml, test_xml] = [ET.parse(set).getroot() for set in [train_path, test_path]]

    [train, test] = [_extract_data_from_xml(xml) for xml in [train_xml, test_xml]]

    data = pd.concat([train, test], ignore_index=True)

    return data


def _compute_file_names(dataset, subject):
    """
    Compute the name of the files, given dataset and subject
    :param dataset: name of dataset
    :param subject: name of subject
    :return: path to training file, path to testing file
    """
    train_dir = os.path.join(cs.path, "data", dataset, "OhioT1DM-training")
    train_file = subject + "-ws-training.xml"
    train_path = os.path.join(train_dir, train_file)

    test_dir = os.path.join(cs.path, "data", dataset, "OhioT1DM-testing")
    test_file = subject + "-ws-testing.xml"
    test_path = os.path.join(test_dir, test_file)

    return train_path, test_path


def _extract_data_from_xml(xml):
    """
    extract glucose, CHO, and insulin from xml and merge the data
    :param xml:
    :return: dataframe
    """
    glucose_df = _get_glucose_from_xml(xml)
    CHO_df = _get_CHO_from_xml(xml)
    insulin_df = _get_insulin_from_xml(xml)

    df = pd.merge(glucose_df, CHO_df, how="outer", on="datetime")
    df = pd.merge(df, insulin_df, how="outer", on="datetime")
    df = df.sort_values("datetime")

    return df


def _get_field_labels(etree, field_index):
    """
    extract labels from xml tree
    :param etree: etree
    :param field_index:  position of field
    :return:
    """
    return list(etree[field_index][0].attrib.keys())


def _iter_fields(etree, field_index):
    """
    extract columns inside xml tree
    :param etree: tree
    :param field_index: position of columns
    :return:
    """
    for event in etree[field_index].iter("event"):
        yield list(event.attrib.values())


def _get_CHO_from_xml(xml):
    """
    Extract CHO values from xml
    :param xml:
    :return: CHO dataframe
    """
    labels = _get_field_labels(xml, field_index=5)
    CHO = list(_iter_fields(xml, field_index=5))
    CHO_df = pd.DataFrame(data=CHO, columns=labels)
    CHO_df.drop("type", axis=1, inplace=True)
    CHO_df["ts"] = pd.to_datetime(CHO_df["ts"], format="%d-%m-%Y %H:%M:%S")
    CHO_df["carbs"] = CHO_df["carbs"].astype("float")
    CHO_df.rename(columns={'ts': 'datetime', 'carbs': 'CHO'}, inplace=True)
    return CHO_df


def _get_insulin_from_xml(xml):
    """
    Extract insulin values from xml
    :param xml:
    :return: insulin dataframe
    """
    labels = _get_field_labels(xml, field_index=4)
    insulin = list(_iter_fields(xml, field_index=4))
    insulin_df = pd.DataFrame(data=insulin, columns=labels)
    for col in ["ts_end", "type", "bwz_carb_input"]:
        insulin_df.drop(col, axis=1, inplace=True)
    insulin_df["ts_begin"] = pd.to_datetime(insulin_df["ts_begin"], format="%d-%m-%Y %H:%M:%S")
    insulin_df["dose"] = insulin_df["dose"].astype("float")
    insulin_df.rename(columns={'ts_begin': 'datetime', 'dose': 'insulin'}, inplace=True)
    return insulin_df


def _get_glucose_from_xml(xml):
    """
    Extract glucose values from xml
    :param xml:
    :return: glucose dataframe
    """
    labels = _get_field_labels(xml, field_index=0)
    glucose = list(_iter_fields(xml, field_index=0))
    glucose_df = pd.DataFrame(data=glucose, columns=labels)
    glucose_df["ts"] = pd.to_datetime(glucose_df["ts"], format="%d-%m-%Y %H:%M:%S")
    glucose_df["value"] = glucose_df["value"].astype("float")
    glucose_df.rename(columns={'ts': 'datetime', 'value': 'glucose'}, inplace=True)
    return glucose_df
