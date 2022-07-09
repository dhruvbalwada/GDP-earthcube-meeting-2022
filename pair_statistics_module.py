import numpy as np
import xarray as xr
from scipy.spatial.distance import pdist

def retrieve_year(ds, year) -> xr.Dataset:
    '''Subset the dataset for a region in space and time
    
    Args:
        ds: xarray Dataset
        lon: longitude slice of the subregion
        lat: latitude slice of the subregion
        time: tiem slice of the subregion
    
    Returns: 
        ds_subset: Dataset of the subregion
    '''
    
    # define the mask for the 'obs' dimension
    mask = np.ones(ds.dims['obs'], dtype='bool')

    mask &= (ds.coords['time.year'] == year).values   
    # define the mask for the 'traj' dimension using the ID numbers from the masked observation
    mask_id = np.in1d(ds.ID, np.unique(ds.ids[mask]))
    ds_subset = ds.isel(obs=np.where(mask)[0], traj=np.where(mask_id)[0])

    # Make new row_size using counts
    # this only works if the data is arranged in ascending order of id numbers (which is the case here)
    _, row_size = np.unique(ds_subset.ids, 
                     return_counts=True)
    ds_subset['rowsize'] = xr.DataArray(row_size, dims='traj')
    
    return ds_subset.compute()


class structtype():
    pass

def ds2trajstruct(ds, filt_flag=0):
    # Fundction to return separated out trajectories in 
    # a matlab structure like dataformat.
    
    num_traj = len(ds.ID)
    
    trajs = [ structtype() for i in range(num_traj)]
    
    # create the ids of where the trajectories separate out.
    traj_idx = np.insert(np.cumsum(ds.rowsize.values), 0, 0)
    
    for i in range(num_traj):
        trajs[i].lon = ds.lon[traj_idx[i]:traj_idx[i+1]]
        trajs[i].lat = ds.lat[traj_idx[i]:traj_idx[i+1]]
        trajs[i].u   = ds.ve[traj_idx[i]:traj_idx[i+1]]
        trajs[i].v   = ds.vn[traj_idx[i]:traj_idx[i+1]]
        trajs[i].time= ds.time[traj_idx[i]:traj_idx[i+1]]
        if filt_flag == 1:
            trajs[i].u_lp   = ds.u_lp[traj_idx[i]:traj_idx[i+1]]
            trajs[i].v_lp   = ds.v_lp[traj_idx[i]:traj_idx[i+1]]

    return trajs


def get_hours_per_year(year):
    if np.mod(year,4)==0:
        hours_per_year = 24*366
    else:
        hours_per_year = 24*365
        
    return hours_per_year

def year_hours_2_time(year, hours_index):
    time = np.datetime64(str(year), 'Y') + (np.datetime64(hours_index, 'h') - np.datetime64(0, 'h'))
    return time

def trajstruct2posarrays(trajs, year):
    
    num_traj = len(trajs)
    
    hours_per_year = get_hours_per_year(year)

    lon_array = np.NaN*np.ones((num_traj, hours_per_year))
    lat_array = np.NaN*np.ones((num_traj, hours_per_year))
    #u_array = np.NaN*np.ones((num_traj, hours_per_year))
    #v_array = np.NaN*np.ones((num_traj, hours_per_year))
    
    start_year = np.datetime64(str(year) + '-01-01')
    
    for i in range(num_traj):
    
        hour_index = ((trajs[i].time - start_year).astype('int')/1e9/3600).values.astype('int')
        # we do this because there can be gaps in data

        lon_array[i, hour_index] = trajs[i].lon
        lat_array[i, hour_index] = trajs[i].lat
        #u_array[i, hour_index] = trajs[i].u
        #v_array[i, hour_index] = trajs[i].v
        
    return lon_array, lat_array#, u_array, v_array

def trajstruct2velarrays(trajs, year, vel_type='full'):
    
    num_traj = len(trajs)
    
    hours_per_year = get_hours_per_year(year)

    u_array = np.NaN*np.ones((num_traj, hours_per_year))
    v_array = np.NaN*np.ones((num_traj, hours_per_year))
    
    start_year = np.datetime64(str(year) + '-01-01')
    
    for i in range(num_traj):
    
        hour_index = ((trajs[i].time - start_year).astype('int')/1e9/3600).values.astype('int')
        # we do this because there can be gaps in data

        #lon_array[i, hour_index] = trajs[i].lon
        #lat_array[i, hour_index] = trajs[i].lat
        if vel_type == 'full':
            u_array[i, hour_index] = trajs[i].u
            v_array[i, hour_index] = trajs[i].v
        elif vel_type == 'lp':
            u_array[i, hour_index] = trajs[i].u_lp
            v_array[i, hour_index] = trajs[i].v_lp
        elif vel_type == 'hp':
            u_array[i, hour_index] = trajs[i].u - trajs[i].u_lp
            v_array[i, hour_index] = trajs[i].v - trajs[i].v_lp
    return u_array, v_array

## Functions that will be used by pdist
def dist_rx(XI, XJ):
    rx = (XI[0] - XJ[0])*np.cos(np.deg2rad(0.5*(XI[1] + XJ[1])));
    return rx

def dist_ry(XI, XJ):
    ry = (XI[1] - XJ[1]);
    return ry

def dist_du(UI, UJ):
    du = UI - UJ;
    return du


def dist_geo(XI,XJ):
    # XI or XJ are the coordinates in lon-lat of two different points,
    # where  XI(:,1) is lon and XI(:,2) is the lat. 
    # The function computes the distance between the 2 points.
    #print(XI, XJ)
    X = abs(XI[0] - XJ[0]) *np.cos(np.deg2rad(0.5*(XI[1] + XJ[1]))) *111321;
    Y = abs(XI[1] - XJ[1]) *111321;
    dist = np.sqrt(X**2 + Y**2);
    return dist

## 
def trajarrays2timepairs(trajarrays, year, filt_flag=0):
    
    hours_per_year = get_hours_per_year(year)
    #print(hours_per_year)
    
    pairs_time = [structtype() for i in range(hours_per_year)]
    
    for i in range(hours_per_year):
        X = trajarrays['lon_array'][:,i]
        Y = trajarrays['lat_array'][:,i]
        U = trajarrays['u_array'][:,i]
        V = trajarrays['v_array'][:,i]
        
                
        if filt_flag==1:
            U_lp = trajarrays['u_lp_array'][:,i]
            V_lp = trajarrays['v_lp_array'][:,i]
            U_hp = trajarrays['u_hp_array'][:,i]
            V_hp = trajarrays['v_hp_array'][:,i]

        # remove nans
        Y = Y[~np.isnan(X)]
        U = U[~np.isnan(X)]
        V = V[~np.isnan(X)]
        if filt_flag==1:
            U_lp = U_lp[~np.isnan(X)]
            V_lp = V_lp[~np.isnan(X)]
            U_hp = U_hp[~np.isnan(X)]
            V_hp = V_hp[~np.isnan(X)]
            
        X = X[~np.isnan(X)]
        
        

        Xvec = np.concatenate((np.expand_dims(X, 1), np.expand_dims(Y, 1)), axis=1)

        pairs_time[i].dist = pdist(Xvec, dist_geo)


        rx = pdist(Xvec, dist_rx);
        ry = pdist(Xvec, dist_ry);

        magr = np.sqrt(rx**2 + ry**2);

        rx = rx/magr; ry = ry/magr;

        dux = pdist(np.expand_dims(U, 1), dist_du);
        duy = pdist(np.expand_dims(V, 1), dist_du);

        pairs_time[i].dul = dux*rx + duy*ry;
        pairs_time[i].dut = duy*rx - dux*ry;
        
        if filt_flag == 1:
            dux_lp = pdist(np.expand_dims(U_lp, 1), dist_du);
            duy_lp = pdist(np.expand_dims(V_lp, 1), dist_du);

            pairs_time[i].dul_lp = dux_lp*rx + duy_lp*ry;
            pairs_time[i].dut_lp = duy_lp*rx - dux_lp*ry;
            
            dux_hp = pdist(np.expand_dims(U_hp, 1), dist_du);
            duy_hp = pdist(np.expand_dims(V_hp, 1), dist_du);

            pairs_time[i].dul_hp = dux_hp*rx + duy_hp*ry;
            pairs_time[i].dut_hp = duy_hp*rx - dux_hp*ry;
                  
        
        pairs_time[i].time = year_hours_2_time(year, i)
        
    return pairs_time

def timepairs2list(timepairs, year, filt_flag=0):
    dul = np.array([])
    dut = np.array([])
    dist = np.array([])
    time = np.array([]).astype('datetime64')
    if filt_flag==1:
        dul_lp = np.array([])
        dut_lp = np.array([])        
        dul_hp = np.array([])
        dut_hp = np.array([])        
    
    hours_per_year = get_hours_per_year(year)
    
    for i in range(hours_per_year):
        dul = np.append(dul, timepairs[i].dul)
        dut = np.append(dut, timepairs[i].dut)
        dist = np.append(dist, timepairs[i].dist)
        time = np.append(time, timepairs[i].time + np.zeros_like(timepairs[i].dist).astype('long'))
        
        if filt_flag==1:
            dul_lp = np.append(dul_lp, timepairs[i].dul_lp)
            dut_lp = np.append(dut_lp, timepairs[i].dut_lp)
            dul_hp = np.append(dul_hp, timepairs[i].dul_hp)
            dut_hp = np.append(dut_hp, timepairs[i].dut_hp)
            
    if filt_flag==0:
        return dul, dut, dist, time
    elif filt_flag==1:
        return dul, dut, dist, time, dul_lp, dut_lp, dul_hp, dut_hp

def helmholtz_decompose(dist_axis, SF2ll, SF2tt): 
    SF2rr = np.zeros_like(SF2ll)
    SF2dd = np.zeros_like(SF2tt)
    
    mid_dist_axis = 0.5*(dist_axis[0:-1] + dist_axis[1:])
    dr =  dist_axis[1:] - dist_axis[0:-1]
    diff_SF = SF2tt - SF2ll
    mid_diff_du =  0.5*(diff_SF[0:-1] + diff_SF[1:])
    
    SF2rr[0] = SF2tt[0]
    SF2dd[0] = SF2ll[0]
    
    for i in range(1, len(dist_axis)):
        SF2rr[i] = SF2tt[i] + np.nansum(mid_diff_du[:i]/mid_dist_axis[:i]*dr[:i])
        SF2dd[i] = SF2ll[i] - np.nansum(mid_diff_du[:i]/mid_dist_axis[:i]*dr[:i])
        
    return SF2rr, SF2dd