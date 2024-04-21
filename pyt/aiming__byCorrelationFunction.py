import json, re
import numpy as np

# ========================================================= #
# ===  aiming__byCorrelationFunction.py                 === #
# ========================================================= #

def aiming__byCorrelationFunction():
    
    x1_, x2_, y_     = 0, 1, 2
    configFile       = "cnf/settings.jsonc"
    
    # ------------------------------------------------- #
    # --- [1] declear function                      --- #
    # ------------------------------------------------- #
    def photonFlux_Func( x, dx, sigma ):
        ret = 1.0/( np.sqrt( 2.0*np.pi ) ) * np.exp( - ( x-dx )**2 / (0.5*sigma**2) )
        return( ret )
    
    def distribute_Func( x, Data ):
        stack   = []
        x1Array = Data[:,x1_]
        x2Array = Data[:,x2_]
        yvArray = Data[:,y_ ]
        ret     = np.zeros_like( x )
        for ik,xval in enumerate(x):
            if   ( xval < x1Array[ 0] ):
                ret[ik] = 0.0
            elif ( xval > x1Array[-1] ):
                ret[ik] = 0.0
            else:
                for idx in range( x1Array.shape[0] ):
                    if ( ( xval >= x1Array[idx] ) and ( xval < x2Array[idx] ) ):
                        ret[ik] = yvArray[idx]
                        break
        return( ret )
        
    # ------------------------------------------------- #
    # --- [2] load data & normalization             --- #
    # ------------------------------------------------- #
    #  -- [2-1] load config                         --  #
    with open( configFile, "r" ) as f:
        text   = re.sub(r'/\*[\s\S]*?\*/|//.*', '', f.read() )
        params = json.loads( text )

    #  -- [2-2] load data                           --  #
    with open( params["dist.inpFile"], "r" ) as f:
        dist   = np.loadtxt( f )
        
    #  -- [2-3] normalization                       --  #
    dist[:,y_] = dist[:,y_] / np.sum( dist[:,y_] )
    xMin, xMax = np.min( dist[:,x1_] ), np.max( dist[:,x2_] )
    
    # ------------------------------------------------- #
    # --- [3] axis making                           --- #
    # ------------------------------------------------- #
    min_,max_,num_ = 0, 1, 2
    xAxis          = np.linspace( params["xA.MinMaxNum"][min_], params["xA.MinMaxNum"][max_], \
                                  int(params["xA.MinMaxNum"][num_]) )
    dx_array       = np.linspace( params["xA.MinMaxNum"][min_], params["xA.MinMaxNum"][max_], \
                                  int(params["xA.MinMaxNum"][num_]) )
    dl             = ( params["xA.MinMaxNum"][max_] - params["xA.MinMaxNum"][min_] ) / params["xA.MinMaxNum"][num_]
    stack          = []
    for ik,dx in enumerate(dx_array):
        pFlux      = photonFlux_Func( xAxis, dx, params["gamma.sigma"] )
        distr      = distribute_Func( xAxis, dist )
        innerProd  = np.dot( pFlux, distr ) * dl
        stack     += [ innerProd ]
    innerProds = np.array( stack )
    max_index  = np.argmax( innerProds )
    xMax       =      xAxis[ max_index ]
    pMax       = innerProds[ max_index ]
    nD         = dist.shape[0]
    xArray     = np.insert( np.reshape( dist[:,x1_:x2_+1], (-1,) ), [0,nD*2], [ dist[0,x1_], dist[-1,x2_] ] )
    yArray     = np.insert( np.repeat(  dist[:,y_], 2 )           , [0,nD*2], [ 0.0, 0.0 ] )
    print( "[aiming__byCorrelationFunction.py] xMax   :: {}".format( xMax ) )
    print( "[aiming__byCorrelationFunction.py] max(C) :: {}".format( pMax ) )
    
    # ------------------------------------------------- #
    # --- [4] plot data                             --- #
    # ------------------------------------------------- #
    import nkUtilities.plot1D         as pl1
    import nkUtilities.load__config   as lcf
    import nkUtilities.configSettings as cfs
    pngFile                  = params["pngFile"]
    config                   = lcf.load__config()
    config                   = cfs.configSettings( configType="plot.def", config=config )
    config["FigSize"]        = (4.5,4.5)
    config["plt_position"]   = [ 0.16, 0.16, 0.94, 0.94 ]
    config["plt_xAutoRange"] = False
    config["plt_yAutoRange"] = False
    config["plt_xRange"]     = [ params["plot.xMinMaxNum"][min_], params["plot.xMinMaxNum"][max_] ]
    config["plt_yRange"]     = [ params["plot.yMinMaxNum"][min_], params["plot.yMinMaxNum"][max_] ]
    config["xMajor_Nticks"]  = int( params["plot.xMinMaxNum"][num_] )
    config["yMajor_Nticks"]  = int( params["plot.yMinMaxNum"][num_] )
    config["plt_marker"]     = "o"
    config["plt_markersize"] = 3.0
    config["plt_linestyle"]  = "-"
    config["plt_linewidth"]  = 2.0
    config["xTitle"]         = "$x, \Delta x$"
    config["yTitle"]         = "$C(\Delta x), \phi_n(x,\Delta x), D_n(x)$"
    pFlux      = photonFlux_Func( xAxis, 0.0, params["gamma.sigma"] )
    fig        = pl1.plot1D( config=config, pngFile=pngFile )
    fig.add__plot( xAxis=xAxis   , yAxis=pFlux     , label="$\phi_n(x,\Delta x)$"   )
    fig.add__plot( xAxis=xArray  , yAxis=yArray    , label="$D_n(x)$"      )
    fig.add__plot( xAxis=dx_array, yAxis=innerProds, label="$C(\Delta x)$" )
    fig.add__legend()
    fig.set__axis()
    fig.save__figure()
    return()


# ========================================================= #
# ===   Execution of Pragram                            === #
# ========================================================= #

if ( __name__=="__main__" ):
    aiming__byCorrelationFunction()
