import pickle
import pandas as pd
import time as time


def merge(length):
    start=time.time()

    with open('data/sdss/Nikki_35001-40000.pkl', 'rb') as f:
        x = pickle.load(f)

    df=pd.DataFrame(x)
    # print(x.values)
    # df_dir.head(1)

    with open('data/sdss/direct_sql.pkl', 'rb') as f1:
        x1 = pickle.load(f1)

    df_dir=pd.DataFrame(x1)
    # print(x.values)
    # df.head()
    # print(len(df_dir))

    df_dir["objid"] = df_dir['bestObjID']
    df_dir = df_dir.drop(columns={"bestObjID"})

    ra_list = []
    for i in range(len(df)):
        ra = df["ra"][i][0]
        ra_list.append(ra)
    df["ra"] = ra_list

    dec_list = []
    for i in range(len(df)):
        dec = df["dec"][i][0]
        dec_list.append(dec)
    df["dec"] = dec_list

    z_list = []
    for i in range(len(df)):
        z = df["z"][i][0]
        z_list.append(z)

    df["z"] = z_list

    objid_list = []
    for i in range(len(df)):
        objid = df["objid"][i][0]
        objid_list.append(objid)
    df["objid"] = objid_list

    ws_list = []
    for i in range(len(df)):
        ws = pd.Series(df["wavelength"][i])
        ws_list.append(ws)
    df["wavelength"] = ws_list

    fs_list = []
    for i in range(len(df)):
        fs = pd.Series(df["flux_list"][i])
        fs_list.append(fs)
    df["flux_list"] = fs_list

    df_merge = pd.merge(df, df_dir, on=['objid'])

    df_merge["dec"] = df_merge['dec_y']
    df_merge["z"] = df_merge['z_y']
    df_merge["ra"] = df_merge['ra_y']
    df_merge = df_merge.drop(columns={'dec_x', 'z_x', 'ra_x', 'dec_y', 'z_y', 'ra_y'})

    end=time.time()
    df_merge.to_pickle('data/sdss/FinalTable_40-45.pkl')

    tt=end - start
    print("time:", tt)