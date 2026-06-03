function var = space(varargin)
[var.x_,var.fx_]=domain(0,0);
[var.y_,var.fy_]=domain(0,0);
[var.t_,var.ft_]=domain(0,0);
for index_var =1:3:length(varargin)
    key = varargin{index_var};
    try
        window = varargin{index_var+1};
        sample = varargin{index_var+2};
    catch
        error(['unenougth parm for: ',key])
    end
    
    switch key
        case 'x'
            [var.x_,var.fx_]=domain(window,sample);
        case 'y'
            [var.y_,var.fy_]=domain(window,sample);
%             var.y_=fliplr(var.y_);
        case 't'
            [var.t_,var.ft_]=domain(window,sample);
        otherwise
            error(['undefine key : ',key])
    end
end
[var.x,var.y,var.t] = meshgrid(var.x_,var.y_,var.t_);
[var.fx,var.fy,var.ft] = meshgrid(var.fx_,var.fy_,var.ft_);