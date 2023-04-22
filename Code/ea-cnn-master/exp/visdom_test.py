from visdom import Visdom

viz = Visdom()
viz.line([0.],[0.], win='train_loss', opts=dict(title='train loss'))
# viz.line([loss.item()], [])