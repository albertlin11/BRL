clear;clf;close all;clc

if not(isfolder("RL_fig_prior=0"))
    mkdir RL_fig_prior=0
end
I = readtable("RL_Input_parameter.csv");
for i = 1:height(I)
    clf
    network = string(I{i,1});
    batch_size = string(I{i,2});
    radius = string(I{i,3});


    input = "RL_network=" + network + "_batch_size="+batch_size;
    file = string(input);


    filename = file + '.csv';
    str1 = './data_old_reward_prior=0_2/';
    path = string(str1) + filename;

    A = csvread(path,1,0,[1,0,50,7]);
    A = log10(A);
    y=-A(1:50,8);
    plot(y,'LineWidth',2,'Marker','o');
    xlabel('Timesteps','FontSize',30)
    ylim([-11,-1.5])
    xlim([1,50])
    set(gca,'TickDir','out','FontSize',25, 'LineWidth', 2.0);
    xlabel('Timesteps','FontSize',30)
    if I{i,1} == 0
        ylabel('- log_{10}(Reward)','FontSize',30)
    else
        ylabel('Reward','FontSize',30)
    end
    ylabel('- log_{10}(Reward)','FontSize',30)
    %set(gcf, 'PaperPosition', [0 0 800 540]);
    box off
    ax2 = axes('Position',get(gca,'Position'),...
               'XAxisLocation','top',...
               'YAxisLocation','right',...
               'Color','none',...
               'XColor','k','YColor','k');
    set(ax2,'YTick', []);
    set(ax2,'XTick', []);
    set(gca,'LineWidth',1.5);
    save_filename = './RL_fig_prior=0/' + file + '.png';
    saveas(gcf, save_filename)

end