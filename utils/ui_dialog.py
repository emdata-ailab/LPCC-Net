from traits.api import HasTraits, Instance, Button, on_trait_change
from traitsui.api import View, Item, HSplit, Group
from mayavi import mlab
from mayavi.core.ui.api import MlabSceneModel, SceneEditor


class MyDialog(HasTraits):
    scene_input = Instance(MlabSceneModel, ())
    scene_pred = Instance(MlabSceneModel, ())
    scene_gt = Instance(MlabSceneModel, ())
    button_next_object = Button('Next object')
    button_next_input_pts = Button('Next input pts')

    def __init__(self, dataset):
        # Do not forget to call the parent's __init__
        HasTraits.__init__(self)
        self.dataset = dataset
        self.input_list, self.pred_list, self.gt_list = next(self.dataset)
        self.index_in_current_object = 0
        self.current_object_length = len(self.input_list)
        # set bg colors to pure white
        self.scene_input.mayavi_scene.scene.background = (1.0, 1.0, 1.0)
        self.scene_pred.mayavi_scene.scene.background = (1.0, 1.0, 1.0)
        self.scene_gt.mayavi_scene.scene.background = (1.0, 1.0, 1.0)
        self.show()

    def show(self):
        mlab.clf(figure=self.scene_input.mayavi_scene)
        mlab.points3d(self.input_list[self.index_in_current_object][:, 0],
                      self.input_list[self.index_in_current_object][:, 1],
                      self.input_list[self.index_in_current_object][:, 2],
                      color=(0.12, 0.56, 1.0),
                      scale_factor=0.1,
                      figure=self.scene_input.mayavi_scene)
        mlab.clf(figure=self.scene_pred.mayavi_scene)
        mlab.points3d(self.pred_list[self.index_in_current_object][:, 0],
                      self.pred_list[self.index_in_current_object][:, 1],
                      self.pred_list[self.index_in_current_object][:, 2],
                      color=(0.8, 0.36, 0.36),
                      scale_factor=0.1,
                      figure=self.scene_pred.mayavi_scene)
        mlab.clf(figure=self.scene_gt.mayavi_scene)
        mlab.points3d(self.gt_list[self.index_in_current_object][:, 0],
                      self.gt_list[self.index_in_current_object][:, 1],
                      self.gt_list[self.index_in_current_object][:, 2],
                      color=(0.18, 0.54, 0.34),
                      scale_factor=0.1,
                      figure=self.scene_gt.mayavi_scene)

    @on_trait_change('button_next_object')
    def next_object(self):
        self.input_list, self.pred_list, self.gt_list = next(self.dataset)
        self.index_in_current_object = 0
        self.current_object_length = len(self.input_list)
        self.show()

    @on_trait_change('button_next_input_pts')
    def next_input_pts(self):
        if self.index_in_current_object < self.current_object_length - 1:
            self.index_in_current_object += 1
        else:
            self.index_in_current_object = 0
        self.show()

    # The layout of the dialog created
    view = View(
        Group(
            HSplit(
                Group(
                    Item('scene_input',
                         editor=SceneEditor(),
                         height=450,
                         width=500),
                    show_labels=False,
                ),
                Group(
                    Item('scene_pred',
                         editor=SceneEditor(),
                         height=450,
                         width=500,
                         show_label=False),
                    show_labels=False,
                ),
                Group(
                    Item('scene_gt',
                         editor=SceneEditor(),
                         height=450,
                         width=500,
                         show_label=False),
                    show_labels=False,
                ),
            ),

            Group(
                Item('button_next_object', width=100),
                Item('button_next_input_pts', width=100),
                show_labels=False,
                show_border=True
            ),
        ),
    )


if __name__ == '__main__':
    pass
