import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import r2_score

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from utils import *

np.random.seed(42)
torch.manual_seed(42)


def single_dict(city, period, data_type, war_time_pred, pre_war_time_pred, war_occupancy, pre_war_occupancy):
    return {
        "City": city,
        "Period": period,
        "Filter": data_type,
        "War_time_size": war_time_pred.shape,
        "Pre_War_time_size": pre_war_time_pred.shape,
        "Change_in_occupancy": war_occupancy - pre_war_occupancy,
        "War_Occupancy": war_occupancy,
        "Pre_War_Occupancy": pre_war_occupancy,
        "Min_occupancy": min(war_time_pred['pred_occupancy']),
        "Max_Occupancy": max(war_time_pred['pred_occupancy']),
        "Mean_Occupancy": war_time_pred['pred_occupancy'].mean(),
        "War_Period_Sunday": war_time_pred['pred_day'].value_counts(normalize=True).get('Sunday', 0),
        "War_Period_Saturday": war_time_pred['pred_day'].value_counts(normalize=True).get('Saturday', 0),
        "Pre_War_Period_Sunday": pre_war_time_pred['pred_day'].value_counts(normalize=True).get('Sunday', 0),
        "Pre_War_Period_Saturday": pre_war_time_pred['pred_day'].value_counts(normalize=True).get('Saturday', 0)
    }


def data_output(model_path, data_type, uzhhorod, kyiv, mariupol):
    data = []
    file_name = None
    if data_type == 'all':
        file_name = 'uzhhorod_filtered_with_hist_matching_2020_all_predictions.csv'
        pred = pd.read_csv(f'{model_path}/{file_name}')
    elif data_type == 'selected':
        file_name = 'uzhhorod_filtered_with_hist_matching_2020_predictions.csv'
        pred = pd.read_csv(f'{model_path}/{file_name}')
        pred['parking_lot_name'] = pred['cluster']
        pred = pred.merge(uzhhorod, on='parking_lot_name', how='inner').reset_index(drop=True)

    min_year = pred['date'].min().split('-')[0]
    max_year = pred['date'].max().split('-')[0]

    pred['date'] = pred['image_path'].apply(
        lambda x: datetime.strptime(Path(x).stem[:8], '%Y%m%d').strftime('%Y-%m-%d'))
    pred = pred.sort_values(by='date')
    war_time_pred = pred[(pred['date'] >= f'{max_year}-03-03') & (pred['date'] <= f'{max_year}-03-31')].reset_index(
        drop=True)
    war_occupancy = sum(war_time_pred['pred_occupancy'])
    pre_war_time_pred = pred[(pred['date'] >= f'{min_year}-11-01') & (pred['date'] <= f'{max_year}-02-23')].reset_index(
        drop=True)
    pre_war_occupancy = sum(pre_war_time_pred['pred_occupancy'])
    single_data = single_dict("uzhhorod", "a year before war", "selected_parking", war_time_pred, pre_war_time_pred,
                              war_occupancy, pre_war_occupancy)
    data.append(single_data)

    # print("*"*10 + " uzhhorod close to war" + "*"*10)
    if data_type == 'all':
        file_name = 'uzhhorod_filtered_with_hist_matching_2022_all_predictions.csv'
        pred = pd.read_csv(f'{model_path}/{file_name}')
    elif data_type == 'selected':
        file_name = 'uzhhorod_filtered_with_hist_matching_2022_predictions.csv'
        pred = pd.read_csv(f'{model_path}/{file_name}')
        pred['parking_lot_name'] = pred['cluster']
        pred = pred.merge(uzhhorod, on='parking_lot_name', how='inner').reset_index(drop=True)

    min_year = pred['date'].min().split('-')[0]
    max_year = pred['date'].max().split('-')[0]

    pred['date'] = pred['image_path'].apply(
        lambda x: datetime.strptime(Path(x).stem[:8], '%Y%m%d').strftime('%Y-%m-%d'))
    pred = pred.sort_values(by='date')
    war_time_pred = pred[(pred['date'] >= f'{max_year}-03-03') & (pred['date'] <= f'{max_year}-03-31')].reset_index(
        drop=True)
    war_occupancy = sum(war_time_pred['pred_occupancy'])
    pre_war_time_pred = pred[(pred['date'] >= f'{min_year}-11-01') & (pred['date'] <= f'{max_year}-02-23')].reset_index(
        drop=True)
    pre_war_occupancy = sum(pre_war_time_pred['pred_occupancy'])
    single_data = single_dict("uzhhorod", "close to war", "selected_parking", war_time_pred, pre_war_time_pred,
                              war_occupancy, pre_war_occupancy)
    data.append(single_data)

    # print("*"*10 + " uzhhorod a year into war" + "*"*10)
    if data_type == 'all':
        file_name = 'uzhhorod_filtered_with_hist_matching_2023_all_predictions.csv'
        pred = pd.read_csv(f'{model_path}/{file_name}')
    elif data_type == 'selected':
        file_name = 'uzhhorod_filtered_with_hist_matching_2023_predictions.csv'
        pred = pd.read_csv(f'{model_path}/{file_name}')
        pred['parking_lot_name'] = pred['cluster']
        pred = pred.merge(uzhhorod, on='parking_lot_name', how='inner').reset_index(drop=True)

    min_year = pred['date'].min().split('-')[0]
    max_year = pred['date'].max().split('-')[0]

    pred['date'] = pred['image_path'].apply(
        lambda x: datetime.strptime(Path(x).stem[:8], '%Y%m%d').strftime('%Y-%m-%d'))
    pred = pred.sort_values(by='date')
    war_time_pred = pred[(pred['date'] >= f'2023-03-03') & (pred['date'] <= f'2023-03-31')].reset_index(drop=True)
    war_occupancy = sum(war_time_pred['pred_occupancy'])
    pre_war_time_pred = pred[(pred['date'] >= f'2022-11-01') & (pred['date'] <= f'2023-02-23')].reset_index(drop=True)
    pre_war_occupancy = sum(pre_war_time_pred['pred_occupancy'])
    single_data = single_dict("uzhhorod", "a year into war", "selected_parking", war_time_pred, pre_war_time_pred,
                              war_occupancy, pre_war_occupancy)
    data.append(single_data)

    # print("*"*10 + " Kyiv a year before war" + "*"*10)
    if data_type == 'all':
        file_name = 'kyiv_filtered_with_hist_matching_2020_all_predictions.csv'
        pred = pd.read_csv(f'{model_path}/{file_name}')
    elif data_type == 'selected':
        file_name = 'kyiv_filtered_with_hist_matching_2020_predictions.csv'
        pred = pd.read_csv(f'{model_path}/{file_name}')
        pred['parking_lot_name'] = pred['cluster']
        pred = pred.merge(kyiv, on='parking_lot_name', how='inner').reset_index(drop=True)

    min_year = pred['date'].min().split('-')[0]
    max_year = pred['date'].max().split('-')[0]

    pred['date'] = pred['image_path'].apply(
        lambda x: datetime.strptime(Path(x).stem[:8], '%Y%m%d').strftime('%Y-%m-%d'))
    pred = pred.sort_values(by='date')
    war_time_pred = pred[(pred['date'] >= f'{max_year}-02-24') & (pred['date'] <= f'{max_year}-03-31')].reset_index(
        drop=True)
    war_occupancy = sum(war_time_pred['pred_occupancy'])
    pre_war_time_pred = pred[(pred['date'] >= f'{min_year}-11-01') & (pred['date'] <= f'{max_year}-02-23')].reset_index(
        drop=True)
    pre_war_occupancy = sum(pre_war_time_pred['pred_occupancy'])
    single_data = single_dict("Kyiv", "a year before war", "selected_parking", war_time_pred, pre_war_time_pred,
                              war_occupancy, pre_war_occupancy)
    data.append(single_data)

    # print("*"*10 + " Kyiv close to war" + "*"*10)
    if data_type == 'all':
        file_name = 'kyiv_filtered_with_hist_matching_2022_all_predictions.csv'
        pred = pd.read_csv(f'{model_path}/{file_name}')
    elif data_type == 'selected':
        file_name = 'kyiv_filtered_with_hist_matching_2022_predictions.csv'
        pred = pd.read_csv(f'{model_path}/{file_name}')
        pred['parking_lot_name'] = pred['cluster']
        pred = pred.merge(kyiv, on='parking_lot_name', how='inner').reset_index(drop=True)

    min_year = pred['date'].min().split('-')[0]
    max_year = pred['date'].max().split('-')[0]

    pred['date'] = pred['image_path'].apply(
        lambda x: datetime.strptime(Path(x).stem[:8], '%Y%m%d').strftime('%Y-%m-%d'))
    pred = pred.sort_values(by='date')
    war_time_pred = pred[(pred['date'] >= f'{max_year}-02-24') & (pred['date'] <= f'{max_year}-03-31')].reset_index(
        drop=True)
    war_occupancy = sum(war_time_pred['pred_occupancy'])
    pre_war_time_pred = pred[(pred['date'] >= f'{min_year}-11-01') & (pred['date'] <= f'{max_year}-02-23')].reset_index(
        drop=True)
    pre_war_occupancy = sum(pre_war_time_pred['pred_occupancy'])
    single_data = single_dict("Kyiv", "close to war", "selected_parking", war_time_pred, pre_war_time_pred,
                              war_occupancy, pre_war_occupancy)
    data.append(single_data)

    # print("*"*10 + " Kyiv a year into war" + "*"*10)
    if data_type == 'all':
        file_name = 'kyiv_filtered_with_hist_matching_2023_all_predictions.csv'
        pred = pd.read_csv(f'{model_path}/{file_name}')
    elif data_type == 'selected':
        file_name = 'kyiv_filtered_with_hist_matching_2023_predictions.csv'
        pred = pd.read_csv(f'{model_path}/{file_name}')
        pred['parking_lot_name'] = pred['cluster']
        pred = pred.merge(kyiv, on='parking_lot_name', how='inner').reset_index(drop=True)

    min_year = pred['date'].min().split('-')[0]
    max_year = pred['date'].max().split('-')[0]

    pred['date'] = pred['image_path'].apply(
        lambda x: datetime.strptime(Path(x).stem[:8], '%Y%m%d').strftime('%Y-%m-%d'))
    pred = pred.sort_values(by='date')
    war_time_pred = pred[(pred['date'] >= f'{max_year}-02-24') & (pred['date'] <= f'{max_year}-03-31')].reset_index(
        drop=True)
    war_occupancy = sum(war_time_pred['pred_occupancy'])
    pre_war_time_pred = pred[(pred['date'] >= f'{min_year}-11-01') & (pred['date'] <= f'{max_year}-02-23')].reset_index(
        drop=True)
    pre_war_occupancy = sum(pre_war_time_pred['pred_occupancy'])
    single_data = single_dict("Kyiv", "a year into war", "selected_parking", war_time_pred, pre_war_time_pred,
                              war_occupancy, pre_war_occupancy)
    data.append(single_data)

    # print("*"*10 + " Mariupol a year before war" + "*"*10)
    if data_type == 'all':
        file_name = 'mariupol_filtered_with_hist_matching_2020_all_predictions.csv'
        pred = pd.read_csv(f'{model_path}/{file_name}')
    elif data_type == 'selected':
        file_name = 'mariupol_filtered_with_hist_matching_2020_predictions.csv'
        pred = pd.read_csv(f'{model_path}/{file_name}')
        pred['parking_lot_name'] = pred['cluster']
        pred = pred.merge(mariupol, on='parking_lot_name', how='inner').reset_index(drop=True)

    min_year = pred['date'].min().split('-')[0]
    max_year = pred['date'].max().split('-')[0]

    pred['date'] = pred['image_path'].apply(
        lambda x: datetime.strptime(Path(x).stem[:8], '%Y%m%d').strftime('%Y-%m-%d'))
    pred = pred.sort_values(by='date')
    war_time_pred = pred[(pred['date'] >= f'{max_year}-02-24') & (pred['date'] <= f'{max_year}-03-31')].reset_index(
        drop=True)
    war_occupancy = sum(war_time_pred['pred_occupancy'])
    pre_war_time_pred = pred[(pred['date'] >= f'{min_year}-11-01') & (pred['date'] <= f'{max_year}-02-23')].reset_index(
        drop=True)
    pre_war_occupancy = sum(pre_war_time_pred['pred_occupancy'])
    single_data = single_dict("Mariupol", "a year before war", "selected_parking", war_time_pred, pre_war_time_pred,
                              war_occupancy, pre_war_occupancy)
    data.append(single_data)

    # print("*"*10 + " Mariupol close to war" + "*"*10)
    if data_type == 'all':
        file_name = 'mariupol_filtered_with_hist_matching_2022_all_predictions.csv'
        pred = pd.read_csv(f'{model_path}/{file_name}')
    elif data_type == 'selected':
        file_name = 'mariupol_filtered_with_hist_matching_2022_predictions.csv'
        pred = pd.read_csv(f'{model_path}/{file_name}')
        pred['parking_lot_name'] = pred['cluster']
        pred = pred.merge(mariupol, on='parking_lot_name', how='inner').reset_index(drop=True)

    min_year = pred['date'].min().split('-')[0]
    max_year = pred['date'].max().split('-')[0]

    pred['date'] = pred['image_path'].apply(
        lambda x: datetime.strptime(Path(x).stem[:8], '%Y%m%d').strftime('%Y-%m-%d'))
    pred = pred.sort_values(by='date')
    war_time_pred = pred[(pred['date'] >= f'{max_year}-02-24') & (pred['date'] <= f'{max_year}-03-31')].reset_index(
        drop=True)
    war_occupancy = sum(war_time_pred['pred_occupancy'])
    pre_war_time_pred = pred[(pred['date'] >= f'{min_year}-11-01') & (pred['date'] <= f'{max_year}-02-23')].reset_index(
        drop=True)
    pre_war_occupancy = sum(pre_war_time_pred['pred_occupancy'])
    single_data = single_dict("Mariupol", "close to war", "selected_parking", war_time_pred, pre_war_time_pred,
                              war_occupancy, pre_war_occupancy)
    data.append(single_data)

    # print("*"*10 + " Mariupol a year into war" + "*"*10)
    if data_type == 'all':
        file_name = 'mariupol_filtered_with_hist_matching_2023_all_predictions.csv'
        pred = pd.read_csv(f'{model_path}/{file_name}')
    elif data_type == 'selected':
        file_name = 'mariupol_filtered_with_hist_matching_2023_predictions.csv'
        pred = pd.read_csv(f'{model_path}/{file_name}')
        pred['parking_lot_name'] = pred['cluster']
        pred = pred.merge(mariupol, on='parking_lot_name', how='inner').reset_index(drop=True)

    min_year = pred['date'].min().split('-')[0]
    max_year = pred['date'].max().split('-')[0]

    pred['date'] = pred['image_path'].apply(
        lambda x: datetime.strptime(Path(x).stem[:8], '%Y%m%d').strftime('%Y-%m-%d'))
    pred = pred.sort_values(by='date')
    war_time_pred = pred[(pred['date'] >= f'{max_year}-02-24') & (pred['date'] <= f'{max_year}-03-31')].reset_index(
        drop=True)
    war_occupancy = sum(war_time_pred['pred_occupancy'])
    pre_war_time_pred = pred[(pred['date'] >= f'{min_year}-11-01') & (pred['date'] <= f'{max_year}-02-23')].reset_index(
        drop=True)
    pre_war_occupancy = sum(pre_war_time_pred['pred_occupancy'])
    single_data = single_dict("Mariupol", "a year into war", "selected_parking", war_time_pred, pre_war_time_pred,
                              war_occupancy, pre_war_occupancy)
    data.append(single_data)

    df = pd.DataFrame(data)
    return df


def sari_prediction(model_path, df_rsv):
    sari_bbox = pd.read_csv(f'{model_path}/sari_bbox_predictions.csv')
    sari_pbox = pd.read_csv(f'{model_path}/sari_pbox_predictions.csv')

    sari_bbox = sari_bbox.groupby(['date', 'day'])['pred_occupancy'].mean().reset_index()
    sari_df_bbox = sari_bbox.merge(df_rsv, on='date', how='inner')
    sari_df_bbox.rename(columns={'pred_occupancy': 'pred_occupancy_bbox'}, inplace=True)

    sari_pbox = sari_pbox.groupby(['date', 'day'])['pred_occupancy'].mean().reset_index()
    sari_df_pbox = sari_pbox.merge(df_rsv, on='date', how='inner')
    sari_df_pbox.rename(columns={'pred_occupancy': 'pred_occupancy_pbox'}, inplace=True)

    sari_df_pbox['Id'] = sari_df_pbox.apply(lambda x: f'{x.date}_{x.day}', axis=1)
    sari_df_bbox['Id'] = sari_df_bbox.apply(lambda x: f'{x.date}_{x.day}', axis=1)
    sari_df_bbox.drop(columns=['day', 'date', 'Rate'], inplace=True)

    sari_df = sari_df_pbox.merge(sari_df_bbox, on='Id', how='inner')

    fig, ax1 = plt.subplots()

    ax1.plot(sari_df['date'], sari_df['pred_occupancy_bbox'], color='b', label='Bounding box')
    ax1.plot(sari_df['date'], sari_df['pred_occupancy_pbox'], color='orange', label='Manually drawn')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Predicted Occupancy', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.tick_params(axis='x', rotation=45)

    ax2 = ax1.twinx()
    ax2.plot(sari_df['date'], sari_df['Rate'], color='r')
    ax2.set_ylabel('Hospitalisation Rate', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.show()

    print(f"R^2 for bbox: {r2_score(sari_df['Rate'], sari_df['pred_occupancy_bbox'])}")
    print(f"R^2 for pbox: {r2_score(sari_df['Rate'], sari_df['pred_occupancy_pbox'])}")
    print(f"R^2 between {r2_score(sari_df['pred_occupancy_bbox'], sari_df['pred_occupancy_pbox'])}")
    print(
        f"R^2 for both {r2_score(sari_df['Rate'], sari_df[['pred_occupancy_bbox', 'pred_occupancy_pbox']].mean(axis=1))}")
    return sari_df[['pred_occupancy_bbox', 'pred_occupancy_pbox', 'Rate']].corr()


def dynamic_rolling_mean(data, max_window):
    series = pd.Series(data)
    rolling_means = []

    for i in range(len(series)):
        window_size = min(max_window, i + 1)

        window_mean = series.iloc[max(0, i - window_size + 1):i + 1].mean()

        rolling_means.append(window_mean)

    return pd.Series(rolling_means)


def plot_of_monthly_occ(path_to_data, place='uzz', data_type='all'):
    # path_to_data = path_to_data.replace('../../', '')
    file_name = None
    if place != 'all':
        if place == 'uzz':
            city = "Uzzhorrod"

            if data_type == 'all':
                file_name = 'uzhhorod_filtered_with_hist_matching_2022_all_predictions'
            elif data_type == 'selected':
                file_name = 'uzhhorod_filtered_with_hist_matching_2022_predictions'

            sample_data = pd.read_csv(f'{path_to_data}/{file_name}.csv')
            sample_data = sample_data[(sample_data['date'] >= '2021-10-01') & (sample_data['date'] <= '2022-06-31')]
            sample_data = sample_data.groupby(['date']).agg({'pred_occupancy': 'mean'}).reset_index()
            sample_data.sort_values(by='date', inplace=True)
            war_period = sample_data[(sample_data['date'] >= '2022-03-03') & (sample_data['date'] <= '2022-03-31')]

        elif place == 'ky':
            city = "Kyiv"
            if data_type == 'all':
                file_name = 'kyiv_filtered_with_hist_matching_2022_all_predictions'
            elif data_type == 'selected':
                file_name = 'kyiv_filtered_with_hist_matching_2022_predictions'

            sample_data = pd.read_csv(f'{path_to_data}/{file_name}.csv')
            sample_data = sample_data[(sample_data['date'] >= '2021-10-01') & (sample_data['date'] <= '2022-06-31')]
            sample_data = sample_data.groupby(['date']).agg({'pred_occupancy': 'mean'}).reset_index()
            sample_data.sort_values(by='date', inplace=True)
            war_period = sample_data[(sample_data['date'] >= '2022-02-24') & (sample_data['date'] <= '2022-03-31')]

        elif place == 'mar':
            city = "Mariupol"
            if data_type == 'all':
                file_name = 'mariupol_filtered_with_hist_matching_2022_all_predictions'
            elif data_type == 'selected':
                file_name = 'mariupol_filtered_with_hist_matching_2022_predictions'
            sample_data = pd.read_csv(f'{path_to_data}/{file_name}.csv')
            sample_data = sample_data[(sample_data['date'] >= '2021-10-01') & (sample_data['date'] <= '2022-06-31')]
            sample_data = sample_data.groupby(['date']).agg({'pred_occupancy': 'mean'}).reset_index()
            sample_data.sort_values(by='date', inplace=True)
            war_period = sample_data[(sample_data['date'] >= '2022-02-24') & (sample_data['date'] <= '2022-03-31')]

        sample_data['date'] = pd.to_datetime(sample_data['date'])
        sample_data.sort_values(by='date', inplace=True)
        sample_data['year_month'] = sample_data['date'].dt.to_period('M')
        sample_data = sample_data.groupby(['year_month']).agg({'pred_occupancy': 'mean'}).reset_index()
        sample_data.rename(columns={'year_month': 'date'}, inplace=True)
        sample_data['date'] = sample_data['date'].astype(str)

        plt.figure(figsize=(10, 5))

        # Highlight March 2022 with a shaded region
        if place == 'uzz':
            plt.axvspan(pd.to_datetime('2022-03-03'), pd.to_datetime('2022-03-31'), color='red', alpha=0.3,
                        label='Invasion Period')
        else:
            plt.axvspan(pd.to_datetime('2022-02-24'), pd.to_datetime('2022-03-31'), color='red', alpha=0.3,
                        label='Invasion Period')

        plt.plot(pd.to_datetime(sample_data['date']), sample_data['pred_occupancy'], label='Monthly Occupancy',
                 color='blue', linestyle='dashed', alpha=0.4)
        # plt.fill_between(sample_data['date'], sample_data['pred_occupancy'], color='blue', alpha=0.2)
        plt.xticks(rotation=45)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Average Occupancy')
        plt.title(f'Average Occupancy in {city} close to war')
        plt.legend()
        plt.show()

    else:
        city = "All"
        if data_type == 'all':
            ky_data = pd.read_csv(f'{path_to_data}/kyiv_filtered_with_hist_matching_2022_all_predictions.csv')
            uzz_data = pd.read_csv(f'{path_to_data}/uzhhorod_filtered_with_hist_matching_2022_all_predictions.csv')
            mar_data = pd.read_csv(f'{path_to_data}/mariupol_filtered_with_hist_matching_2022_all_predictions.csv')

        elif data_type == 'selected':
            ky_data = pd.read_csv(f'{path_to_data}/kyiv_filtered_with_hist_matching_2022_predictions.csv')
            # ky_data['city'] = 'Kyiv'
            uzz_data = pd.read_csv(f'{path_to_data}/uzhhorod_filtered_with_hist_matching_2022_predictions.csv')
            # uzz_data['city'] = 'Uzzhorrod'
            mar_data = pd.read_csv(f'{path_to_data}/mariupol_filtered_with_hist_matching_2022_predictions.csv')
            # mar_data['city'] = 'Mariupol'

        ky_data = ky_data[(ky_data['date'] >= '2021-10-01') & (ky_data['date'] <= '2022-06-31')]
        ky_data = ky_data.groupby(['date']).agg({'pred_occupancy': 'mean'}).reset_index()
        ky_data.sort_values(by='date', inplace=True)

        ky_data['date'] = pd.to_datetime(ky_data['date'])
        ky_data.sort_values(by='date', inplace=True)
        ky_data['year_month'] = ky_data['date'].dt.to_period('M')
        ky_data = ky_data.groupby(['year_month']).agg({'pred_occupancy': 'mean'}).reset_index()
        ky_data.rename(columns={'year_month': 'date'}, inplace=True)
        ky_data['date'] = ky_data['date'].astype(str)
        ky_data['city'] = 'Kyiv'

        uzz_data = uzz_data[(uzz_data['date'] >= '2021-10-01') & (uzz_data['date'] <= '2022-06-31')]
        uzz_data = uzz_data.groupby(['date']).agg({'pred_occupancy': 'mean'}).reset_index()
        uzz_data.sort_values(by='date', inplace=True)
        # war_period = uzz_data[(uzz_data['date'] >= '2022-02-24') & (uzz_data['date'] <= '2022-03-31')]
        uzz_data['date'] = pd.to_datetime(uzz_data['date'])
        uzz_data.sort_values(by='date', inplace=True)
        uzz_data['year_month'] = uzz_data['date'].dt.to_period('M')
        uzz_data = uzz_data.groupby(['year_month']).agg({'pred_occupancy': 'mean'}).reset_index()
        uzz_data.rename(columns={'year_month': 'date'}, inplace=True)
        uzz_data['date'] = uzz_data['date'].astype(str)
        uzz_data['city'] = 'Uzzhorrod'

        mar_data = mar_data[(mar_data['date'] >= '2021-10-01') & (mar_data['date'] <= '2022-06-31')]
        mar_data = mar_data.groupby(['date']).agg({'pred_occupancy': 'mean'}).reset_index()
        mar_data.sort_values(by='date', inplace=True)
        # war_period = mar_data[(mar_data['date'] >= '2022-02-24') & (mar_data['date'] <= '2022-03-31')]

        mar_data['date'] = pd.to_datetime(mar_data['date'])
        mar_data.sort_values(by='date', inplace=True)
        mar_data['year_month'] = mar_data['date'].dt.to_period('M')
        mar_data = mar_data.groupby(['year_month']).agg({'pred_occupancy': 'mean'}).reset_index()
        mar_data.rename(columns={'year_month': 'date'}, inplace=True)
        mar_data['date'] = mar_data['date'].astype(str)
        mar_data['city'] = 'Mariupol'

        # sample_data = pd.concat([ky_data, uzz_data, mar_data])

        plt.figure(figsize=(10, 5))
        plt.axvspan(pd.to_datetime('2022-02-24'), pd.to_datetime('2022-03-31'), color='red', alpha=0.3,
                    label='Invasion Period')
        for data in [ky_data, uzz_data, mar_data]:
            city = data['city'].unique()[0]
            plt.plot(pd.to_datetime(data['date']), data['pred_occupancy'], label=f'{city}', linestyle='dashed', alpha=1)

        plt.xticks(rotation=45)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Average Occupancy')
        # plt.title(f'Average Occupancy in {city} close to war')
        plt.legend()
        # Format the x-axis labels
        current_labels = data['date'].unique()
        # print(current_labels)
        plt.gca().set_xticklabels([pd.to_datetime(label).strftime('%Y-%b') for label in current_labels])

        plt.show()


def plot_monthly_agg(df, path_to_file, data_type='all'):
    close_to_war = df[df.Period == 'close to war']
    close_to_war['invasion'] = close_to_war.apply(lambda x: x.War_Occupancy / (x.War_Occupancy + x.Pre_War_Occupancy),
                                                  axis=1)
    close_to_war['pre_invasion'] = close_to_war.apply(
        lambda x: x.Pre_War_Occupancy / (x.War_Occupancy + x.Pre_War_Occupancy), axis=1)
    close_to_war['City'] = close_to_war['City'].apply(lambda x: x.title())
    close_to_war = close_to_war.melt(id_vars=['City', 'Filter', 'War_Occupancy', 'Pre_War_Occupancy'],
                                     value_vars=['invasion', 'pre_invasion'], var_name='Period', value_name='Occupancy')
    close_to_war.sort_values(by='Occupancy', ascending=True, inplace=True)
    sns.catplot(data=close_to_war, kind="bar", x="City", y="Occupancy", hue="Period", ci="sd", palette="dark", alpha=.6,
                height=6)
    # plot_of_monthly_occ(path_to_file, 'uzz', data_type)
    # plot_of_monthly_occ(path_to_file, 'ky', data_type)
    # plot_of_monthly_occ(path_to_file, 'mar', data_type)
    plot_of_monthly_occ(path_to_file, 'all', data_type)


def min_max_normalize(values):
    min_val = np.min(values)
    max_val = np.max(values)
    return (values - min_val) / (max_val - min_val)


def plot_for_cities(df):
    # Extract the periods
    periods = df['Period'].unique()

    normalized_data = df.copy()
    for period in periods:
        mask = df['Period'] == period
        values = df[mask]['Occupancy']
        normalized_values = min_max_normalize(values)
        normalized_data.loc[mask, 'Occupancy'] = normalized_values

    df = normalized_data.copy()
    uzz = df[df.City == 'Uzzhorrod']
    ky = df[df.City == 'Kyiv']
    mar = df[df.City == 'Mariupol']

    plt.figure(figsize=(10, 6))
    plt.subplot(1, 3, 1)
    plt.title('Uzzhorrod')

    plt.plot(uzz['Period'], uzz['War_Occupancy'], '-o', label='War Period')
    plt.plot(uzz['Period'], uzz['Pre_War_Occupancy'], '-o', label='Pre War Period')
    plt.xlabel('Period')
    plt.ylabel('Occupancy')
    plt.xticks(rotation=45)
    plt.legend()
    plt.subplot(1, 3, 2)
    plt.title('Kyiv')
    plt.plot(ky['Period'], ky['War_Occupancy'], '-o', label='War Period')
    plt.plot(ky['Period'], ky['Pre_War_Occupancy'], '-o', label='Pre War Period')
    plt.xlabel('Period')
    plt.ylabel('Occupancy')
    plt.xticks(rotation=45)
    plt.legend()
    plt.subplot(1, 3, 3)
    plt.title('Mariupol')
    plt.plot(mar['Period'], mar['War_Occupancy'], '-o', label='War Period')
    plt.plot(mar['Period'], mar['Pre_War_Occupancy'], '-o', label='Pre War Period')
    plt.xlabel('Period')
    plt.ylabel('Occupancy')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()
