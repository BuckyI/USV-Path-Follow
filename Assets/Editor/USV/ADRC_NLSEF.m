function [sys,x0,str,ts,simStateCompliance] = ADRC_NLSEF(t,x,u,flag,para)
switch flag
  % Initialization %
  case 0
    [sys,x0,str,ts,simStateCompliance]=mdlInitializeSizes(para);
  % Derivatives %
  case 1
    sys=mdlDerivatives(t,x,u);
  % Update %
  case 2
    sys=mdlUpdate(t,x,u,para);
  % Outputs %
  case 3
    sys=mdlOutputs(t,x,u);
  % GetTimeOfNextVarHit %
  case 4
    sys=mdlGetTimeOfNextVarHit(t,x,u);
  % Terminate %
  case 9
    sys=mdlTerminate(t,x,u);
  % Unexpected flags %
  otherwise
    DAStudio.error('Simulink:blocks:unhandledFlag', num2str(flag));
end

function [sys,x0,str,ts,simStateCompliance]=mdlInitializeSizes(para)

sizes = simsizes;

sizes.NumContStates  = 0;
sizes.NumDiscStates  = 1;
sizes.NumOutputs     = 1;  % NOTE: u0
sizes.NumInputs      = 2;  % NOTE: e1, e2
sizes.DirFeedthrough = 1;
sizes.NumSampleTimes = 1;   % at least one sample time is needed

sys = simsizes(sizes);
% initialize the initial conditions
x0  = [0];
% str is always an empty matrix
str = [];
% initialize the array of sample times
ts  = [para.h 0]; % NOTE: 定采样步长

% Specify the block simStateCompliance. The allowed values are:
%    'UnknownSimState', < The default setting; warn and assume DefaultSimState
%    'DefaultSimState', < Same sim state as a built-in block
%    'HasNoSimState',   < No sim state
%    'DisallowSimState' < Error out when saving or restoring the model sim state
simStateCompliance = 'UnknownSimState';


function sys=mdlDerivatives(t,x,u)

sys = [];

function sys=mdlUpdate(t,x,u,para)
h=para.h; % step
% u0 = alpha1*fal(e1,a1,h)+alpha2*fal(e2,a2,h);
u0 = para.beta1*fal(u(1),para.a1,h)+para.beta2*fal(u(2),para.a2,h);
sys = u0;

function fa=fal(x,a,delta) % fal function for adrc
if abs(x)<=delta
    fa=x/(delta^(1-a));
else
    fa=sign(x)*abs(x)^a;
end


function sys=mdlOutputs(t,x,u)
sys = x;

function sys=mdlGetTimeOfNextVarHit(t,x,u)

sampleTime = 1;    %  Example, set the next hit to be one second later.
sys = t + sampleTime;

function sys=mdlTerminate(t,x,u)

sys = [];


% 此函数并没有被引用, 在simulink中 MATLAB Function 中可以使用下面的代码
% 为了格式统一性.. 最后还是使用 S-Function 实现
% function u0 = ADRC_NLSEF(e1,e2)
% globle para
% h=para.h;
% alpha1=para.alpha1;
% alpha2=para.alpha2;
% a1=para.a1;
% a2=para.a2;
% 
% u0 = alpha1*fal(e1,a1,h)+alpha2*fal(e2,a2,h);
% % 按书上, fal 的 delta 参数应该为 5~10 h, 这里有待商榷 TODO
% 
% function fa=fal(x,a,delta) % fal function for adrc
%     if abs(x)<=delta
%         fa=x/(delta^(1-a));
%     else
%         fa=sign(x)*abs(x)^a;
%     end
