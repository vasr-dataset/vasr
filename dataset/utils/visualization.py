import io
from copy import deepcopy

import PIL
import json
import matplotlib.pyplot as plt
import matplotlib as mpl
import cv2
import random
import os
import numpy as np
from PIL import Image

from dataset.config import swig_images_path, VIS_ENLARGE_FACTOR, NUM_CANDIDATES, FONT_SIZE, \
    get_difference


def visualize_pair(AB_match_dict, out_p=None, plot_annotations=True):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15,8))
    A_data = AB_match_dict['A_data']
    B_data = AB_match_dict['B_data']
    A_img = cv2.imread(os.path.join(swig_images_path, A_data['A_img']))[:,:,::-1]
    B_img = cv2.imread(os.path.join(swig_images_path, B_data['B_img']))[:,:,::-1]
    axs[0].imshow(A_img)
    axs[1].imshow(B_img)
    if plot_annotations:
        A_str_first = {k:v[0] for k,v in A_data['A_str'].items()}
        B_str_first = {k:v[0] for k,v in B_data['B_str'].items()}
        axs[0].set_title(analogy2str(A_str_first, AB_match_dict['A_verb'], B_data['different_key']), fontsize=FONT_SIZE)
        axs[1].set_title(analogy2str(B_str_first, AB_match_dict['B_verb'], B_data['different_key']), fontsize=FONT_SIZE)
    diff_item_A, diff_item_B, different_key = get_difference(AB_match_dict, A_data, B_data, str_fmt=True)
    plt.suptitle(f"difference: {different_key}, {diff_item_A}->{diff_item_B}", fontsize=22)
    plt.tight_layout()
    if out_p:
        plt.savefig(out_p)
        plt.close(fig)
        plt.cla()
    else:
        plt.show()


def visualize_analogy(analogy, out_p=None, plot_annotations=False, return_fig=False, hide_answer=False):
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(5 * VIS_ENLARGE_FACTOR, 5 * VIS_ENLARGE_FACTOR))
    A_img = cv2.imread(os.path.join(swig_images_path, analogy['A_img']))[:,:,::-1]
    B_img = cv2.imread(os.path.join(swig_images_path, analogy['B_img']))[:,:,::-1]
    axs[0][0].imshow(A_img)
    title_font_size = 12
    axs[0][0].set_title("A", fontsize=title_font_size * VIS_ENLARGE_FACTOR)
    axs[0][1].imshow(B_img)
    axs[0][1].set_title("A'", fontsize=title_font_size * VIS_ENLARGE_FACTOR)
    C_img = cv2.imread(os.path.join(swig_images_path, analogy['C_img']))[:,:,::-1]
    D_img = cv2.imread(os.path.join(swig_images_path, analogy['D_img']))[:,:,::-1]
    axs[1][0].imshow(C_img)
    axs[1][0].set_title("B", fontsize=title_font_size * VIS_ENLARGE_FACTOR)
    if hide_answer:
        question_mark_image = cv2.imread(os.path.join('..', 'assets', 'qmark.png'))
        question_mark_image_resized = cv2.resize(question_mark_image, (256,256), interpolation=cv2.INTER_AREA)
        axs[1][1].imshow(question_mark_image_resized)
        # white_image = np.ones(C_img.shape)
    else:
        axs[1][1].imshow(D_img)
    axs[1][1].set_title("B'", fontsize=title_font_size * VIS_ENLARGE_FACTOR)
    if plot_annotations:
        annotations_cols = ['A_annotations_str', 'B_annotations_str', 'C_annotations_str', 'D_annotations_str']
        for c in annotations_cols:
            analogy[f'{c}_first'] = {k: v[0] if v is not None else None for k,v in analogy[c].items()}
        fontsize = 12
        axs[0][0].set_title(analogy2str(analogy['A_annotations_str_first'], analogy['A_verb'], analogy['different_key']), fontsize=fontsize)
        axs[0][1].set_title(analogy2str(analogy['B_annotations_str_first'], analogy['B_verb'], analogy['different_key']), fontsize=fontsize)
        axs[1][0].set_title(analogy2str(analogy['C_annotations_str_first'], analogy['C_verb'], analogy['different_key']), fontsize=fontsize)
        axs[1][1].set_title(analogy2str(analogy['D_annotations_str_first'], analogy['D_verb'], analogy['different_key']), fontsize=fontsize)
        plt.suptitle(f"difference: {analogy['different_key']}, {analogy['diff_item_A_str_first']}->{analogy['diff_item_B_str_first']}", fontsize=14)
    else:
        if not hide_answer:
            plt.suptitle(f"difference: {analogy['different_key']}", fontsize=12)
    for ax_idx, ax in enumerate(axs.reshape(-1)):
        ax.grid(False)
        # ax.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        # for tic in ax.xaxis.get_major_ticks():
        #     tic.tick1On = tic.tick2On = False
        # for tick in ax.xaxis.get_major_ticks():
        #     tick.tick1line.set_visible(False)
        #     tick.tick2line.set_visible(False)
        #     tick.label1.set_visible(False)
        #     tick.label2.set_visible(False)

    plt.tight_layout()
    if return_fig:
        return fig
    if out_p:
        plt.savefig(out_p)
        plt.close(fig)
        plt.cla()
        print(out_p)
    else:
        plt.show()

def get_img_from_fig(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def visualize_analogy_and_distractors(analogy, out_p=None, show_analogy_type=False, show_analogy_answer=False, show_bbox=False):
    fig = plt.figure(constrained_layout=True, figsize=(20, 11))
    titles_size = 25
    labels_size = 18
    subfigs = fig.subfigures(2, 1, wspace=0.02, hspace=0.2)
    subfigs[0].suptitle('Analogy', fontsize=titles_size)
    subfigs[1].suptitle('Candidates', fontsize=titles_size)

    axsLeft = subfigs[0].subplots(1, 6, gridspec_kw={'width_ratios': [14, 14, 1, 1, 14, 14]})
    axsRight = subfigs[1].subplots(1, 4)

    for ax_idx, ax in enumerate(axsLeft.reshape(-1)):
        ax.grid(False)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
    for ax_idx, ax in enumerate(axsRight.reshape(-1)):
        ax.grid(False)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)

    if not show_analogy_answer or show_bbox is False:
        A_img = cv2.resize(cv2.imread(os.path.join(swig_images_path, analogy['A_img']))[:, :, ::-1], (512, 512), interpolation = cv2.INTER_AREA)
        B_img = cv2.resize(cv2.imread(os.path.join(swig_images_path, analogy['B_img']))[:, :, ::-1], (512, 512), interpolation = cv2.INTER_AREA)
        C_img = cv2.resize(cv2.imread(os.path.join(swig_images_path, analogy['C_img']))[:, :, ::-1], (512, 512), interpolation = cv2.INTER_AREA)
        D_img = cv2.resize(cv2.imread(os.path.join(swig_images_path, analogy['D_img']))[:, :, ::-1], (512, 512), interpolation = cv2.INTER_AREA)
    else:
        A_img = get_image_with_bbox_resized(analogy, 'A_img', 'A_bounding_box')
        B_img = get_image_with_bbox_resized(analogy, 'B_img', 'B_bounding_box')
        C_img = get_image_with_bbox_resized(analogy, 'C_img', 'C_bounding_box')
        D_img = get_image_with_bbox_resized(analogy, 'D_img', 'D_bounding_box')

    axsLeft[0].imshow(A_img)
    axsLeft[0].set_title("A", fontsize=labels_size)
    axsLeft[1].imshow(B_img)
    axsLeft[1].set_title("A'", fontsize=labels_size)
    axsLeft[4].imshow(C_img)
    axsLeft[4].set_title("B", fontsize=labels_size)
    separator = cv2.imread(os.path.join('..', 'assets', 'analogy-separator.jpg'))
    axsLeft[2].imshow(separator)
    axsLeft[2].axis('off')
    axsLeft[3].imshow(separator)
    axsLeft[3].axis('off')
    if not show_analogy_answer:
        question_mark_image = cv2.imread(os.path.join('..', 'assets', 'qmark.png'))
        question_mark_image_resized = cv2.resize(question_mark_image, (256, 256), interpolation=cv2.INTER_AREA)
        axsLeft[5].imshow(question_mark_image_resized)
        # white_image = np.ones(C_img.shape)
    else:
        axsLeft[5].imshow(D_img)
    axsLeft[5].set_title("B'", fontsize=labels_size)
    if show_analogy_type:
        if show_analogy_answer:
            plt.suptitle(
                f"difference: {analogy['different_key']}, {analogy['diff_item_A_str_first']}->{analogy['diff_item_B_str_first']}",
                fontsize=14 * VIS_ENLARGE_FACTOR)
        else:
            plt.suptitle(f"difference: {analogy['different_key']}", fontsize=14 * VIS_ENLARGE_FACTOR)

    for c_idx, c in enumerate(analogy['candidates_images']):
        c_img = cv2.imread(os.path.join(swig_images_path, c))[:, :, ::-1]
        relevant_ax = axsRight[c_idx]
        relevant_ax.imshow(c_img)
        relevant_ax.set_title(f"{c_idx + 1}", fontsize=labels_size)
    if out_p:
        plt.savefig(out_p)
        print(f"Wrote item to {out_p}")
        plt.close(fig)
        plt.cla()
    else:
        plt.show()


def get_image_with_bbox_resized(analogy, img_key, bbox_key):
    img_before_resize = cv2.imread(os.path.join(swig_images_path, analogy[img_key]))
    d_bbox = analogy[bbox_key]
    if not type(d_bbox) == dict:
        d_bbox = json.loads(d_bbox.replace("'",'"'))
    if analogy['different_key'] != 'verb':
        x1, y1, x2, y2 = d_bbox[analogy['different_key']]
        img_before_resize = cv2.rectangle(img_before_resize, (x1, y1), (x2, y2), (0, 0, 255), 3)
    img_rect_resized = cv2.resize(img_before_resize, (512, 512), interpolation = cv2.INTER_AREA)[:,:,::-1]
    return img_rect_resized


def working_4x2_grid(analogy, out_p, show_analogy_answer, show_analogy_type):
    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))
    A_img = cv2.imread(os.path.join(swig_images_path, analogy['A_img']))[:, :, ::-1]
    B_img = cv2.imread(os.path.join(swig_images_path, analogy['B_img']))[:, :, ::-1]
    axs[0][0].imshow(A_img)
    title_font_size = 12
    axs[0][0].set_title("A", fontsize=title_font_size * VIS_ENLARGE_FACTOR)
    axs[0][1].imshow(B_img)
    axs[0][1].set_title("A'", fontsize=title_font_size * VIS_ENLARGE_FACTOR)
    C_img = cv2.imread(os.path.join(swig_images_path, analogy['C_img']))[:, :, ::-1]
    D_img = cv2.imread(os.path.join(swig_images_path, analogy['D_img']))[:, :, ::-1]
    axs[1][0].imshow(C_img)
    axs[1][0].set_title("B", fontsize=title_font_size * VIS_ENLARGE_FACTOR)
    if not show_analogy_answer:
        question_mark_image = cv2.imread(os.path.join('..', 'assets', 'qmark.png'))
        question_mark_image_resized = cv2.resize(question_mark_image, (256, 256), interpolation=cv2.INTER_AREA)
        axs[1][1].imshow(question_mark_image_resized)
        # white_image = np.ones(C_img.shape)
    else:
        axs[1][1].imshow(D_img)
    axs[1][1].set_title("B'", fontsize=title_font_size * VIS_ENLARGE_FACTOR)
    if show_analogy_type:
        if show_analogy_answer:
            plt.suptitle(
                f"difference: {analogy['different_key']}, {analogy['diff_item_A_str_first']}->{analogy['diff_item_B_str_first']}",
                fontsize=14 * VIS_ENLARGE_FACTOR)
        else:
            plt.suptitle(f"difference: {analogy['different_key']}", fontsize=14 * VIS_ENLARGE_FACTOR)
    candidates = deepcopy(analogy['candidates'])
    candidates.pop()
    candidates_images = candidates + [analogy['D_img']]
    assert len(candidates_images) == 4
    random.shuffle(candidates_images)
    idx2loc = {0: (0, 2), 1: (0, 3), 2: (1, 2), 3: (1, 3)}
    for c_idx, c in enumerate(candidates_images):
        c_img = cv2.imread(os.path.join(swig_images_path, c))[:, :, ::-1]
        relevant_ax = axs[idx2loc[c_idx]]
        relevant_ax.imshow(c_img)
        relevant_ax.set_title(f"C{c_idx + 1}", fontsize=title_font_size * VIS_ENLARGE_FACTOR)
        # ax.set_ylabel(f"C{c_idx + 1}", fontsize=10 * VIS_ENLARGE_FACTOR, rotation=0, labelpad=10)
        relevant_ax.grid(False)
        # ax.axis('off')
        relevant_ax.set_xticklabels([])
        relevant_ax.set_yticklabels([])
    for ax_idx, ax in enumerate(axs.reshape(-1)):
        ax.grid(False)
        # ax.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    plt.tight_layout()
    if out_p:
        plt.savefig(out_p)
        print(f"Wrote item to {out_p}")
        plt.close(fig)
        plt.cla()
    else:
        plt.show()


def initial_try(analogy):
    fig_analogies = visualize_analogy(analogy, return_fig=True, hide_answer=True)
    fig_candidates = visualize_candidates_4x4(analogy, return_fig=True)
    backend = mpl.get_backend()
    mpl.use('agg')
    c1 = fig_analogies.canvas
    c2 = fig_candidates.canvas
    c1.draw()
    c2.draw()
    a1 = np.array(c1.buffer_rgba())
    a2 = np.array(c2.buffer_rgba())
    a = np.hstack((a1,a2))
    mpl.use(backend)
    fig,ax = plt.subplots()
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_axis_off()
    ax.matshow(a)


    fig_analogies_pil = PIL.Image.frombytes('RGB',
                                            fig_analogies.canvas.get_width_height(),
                                            fig_analogies.canvas.tostring_rgb())
    fig_candidates_pil = PIL.Image.frombytes('RGB',
                                             fig_candidates.canvas.get_width_height(),
                                             fig_candidates.canvas.tostring_rgb())
    merged_img = get_concat_h(fig_analogies_pil, fig_candidates_pil)
    merged_img_np = np.array(merged_img)
    plt.axis('off')
    plt.grid(b=None)
    plt.tight_layout()
    plt.imshow(merged_img_np)
    plt.show()


def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def visualize_candidates(analogy, return_fig=False):
    fig, axs = plt.subplots(nrows=NUM_CANDIDATES, ncols=1, figsize=(1.25 * VIS_ENLARGE_FACTOR, 5 * VIS_ENLARGE_FACTOR))
    candidates_images = analogy['candidates'] + [analogy['D_img']]
    random.shuffle(candidates_images)
    x = candidates_images.pop()
    for c_idx, c in enumerate(candidates_images):
        c_img = cv2.imread(os.path.join(swig_images_path, c))[:, :, ::-1]
        axs[c_idx].imshow(c_img)
        # axs[c_idx].set_title(f"C{c_idx + 1}", fontsize=8 * VIS_ENLARGE_FACTOR)
        axs[c_idx].set_ylabel(f"C{c_idx + 1}", fontsize=10 * VIS_ENLARGE_FACTOR, rotation=0, labelpad=10)

    for ax in axs:
        ax.grid(False)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    # ax.axis('off')
    plt.tight_layout()
    # plt.show()
    if return_fig:
        return fig
    else:
        plt.show()

def visualize_candidates_4x4(analogy, return_fig=False):
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(5 * VIS_ENLARGE_FACTOR, 5 * VIS_ENLARGE_FACTOR))
    candidates = deepcopy(analogy['candidates'])
    candidates_images = candidates + [analogy['D_img']]
    random.shuffle(candidates_images)
    for c_idx, (c, ax) in enumerate(zip(candidates_images, axs.reshape(-1))):
        c_img = cv2.imread(os.path.join(swig_images_path, c))[:, :, ::-1]
        ax.imshow(c_img)
        ax.set_title(f"C{c_idx + 1}", fontsize=12 * VIS_ENLARGE_FACTOR)
        # ax.set_ylabel(f"C{c_idx + 1}", fontsize=10 * VIS_ENLARGE_FACTOR, rotation=0, labelpad=10)
        ax.grid(False)
        # ax.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    plt.tight_layout()
    # plt.show()
    if return_fig:
        return fig
    else:
        plt.show()


def visualize_analogy_and_distractors_vsomething(analogy, out_p=None, plot_annotations=True):
    fig = plt.figure(figsize=(15,8))

    A_img = cv2.imread(os.path.join(swig_images_path, analogy['A_img']))[:,:,::-1]
    B_img = cv2.imread(os.path.join(swig_images_path, analogy['B_img']))[:,:,::-1]
    C_img = cv2.imread(os.path.join(swig_images_path, analogy['C_img']))[:,:,::-1]
    fig.add_subplot(2, 5, 1)
    plt.imshow(A_img)
    plt.title("A")
    fig.add_subplot(2, 5, 2)
    plt.imshow(B_img)
    plt.title("A'")
    fig.add_subplot(2, 5, 6)
    plt.imshow(C_img)
    plt.title("B")
    last_r_images = analogy['candidates'] + [analogy['D_img']]
    random.shuffle(last_r_images)
    fig.add_subplot(2, 5, 7)
    plt.imshow(np.ones(C_img.shape))
    plt.title("B'")
    for c_idx, c in enumerate(last_r_images):
        c_img = cv2.imread(os.path.join(swig_images_path, c))[:,:,::-1]
        if c_idx <= 2:
            fig.add_subplot(2, 5, 2 + c_idx + 1)
        else:
            fig.add_subplot(2, 5, 5 + c_idx + 1)
        plt.imshow(c_img)
        plt.title(f"C{c_idx+1}")
    plt.tight_layout()
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    plt.grid(True)

    if out_p:
        plt.savefig(out_p)
        plt.close(fig)
        plt.cla()
    else:
        plt.show()

def visualize_analogy_and_distractors_base(analogy, out_p=None, plot_annotations=True):
    fig = plt.figure(figsize=(15,8))

    A_img = cv2.imread(os.path.join(swig_images_path, analogy['A_img']))[:,:,::-1]
    B_img = cv2.imread(os.path.join(swig_images_path, analogy['B_img']))[:,:,::-1]
    C_img = cv2.imread(os.path.join(swig_images_path, analogy['C_img']))[:,:,::-1]
    fig.add_subplot(2, 5, 1)
    plt.imshow(A_img)
    plt.title("A")
    fig.add_subplot(2, 5, 2)
    plt.imshow(B_img)
    plt.title("A'")
    fig.add_subplot(2, 5, 6)
    plt.imshow(C_img)
    plt.title("B")
    last_r_images = analogy['candidates'] + [analogy['D_img']]
    random.shuffle(last_r_images)
    fig.add_subplot(2, 5, 7)
    plt.imshow(np.ones(C_img.shape))
    plt.title("B'")
    for c_idx, c in enumerate(last_r_images):
        c_img = cv2.imread(os.path.join(swig_images_path, c))[:,:,::-1]
        if c_idx <= 2:
            fig.add_subplot(2, 5, 2 + c_idx + 1)
        else:
            fig.add_subplot(2, 5, 5 + c_idx + 1)
        plt.imshow(c_img)
        plt.title(f"C{c_idx+1}")
    plt.tight_layout()
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    plt.grid(True)

    if out_p:
        plt.savefig(out_p)
        plt.close(fig)
        plt.cla()
    else:
        plt.show()


def visualize_analogy_and_distractors_v4(analogy, out_p=None, plot_annotations=True):
    fig = plt.figure(constrained_layout=True, figsize=(15, 8))
    A_img = cv2.imread(os.path.join(swig_images_path, analogy['A_img']))[:,:,::-1]
    B_img = cv2.imread(os.path.join(swig_images_path, analogy['B_img']))[:,:,::-1]
    C_img = cv2.imread(os.path.join(swig_images_path, analogy['C_img']))[:,:,::-1]
    subfigs = fig.subfigures(1, 2, wspace=0.07)
    axsLeft = subfigs[0].subplots(2, 2)
    axsLeft[0][0].imshow(A_img)
    axsLeft[0][0].set_title('A')
    axsLeft[0][1].imshow(B_img)
    axsLeft[0][1].set_title("A'")
    axsLeft[1][0].imshow(C_img)
    axsLeft[1][0].set_title('B')
    axsLeft[1][1].imshow(A_img)
    axsLeft[1][1].set_title("B'")
    last_r_images = analogy['candidates'] + [analogy['D_img']]
    random.shuffle(last_r_images)
    fig.add_subplot(2, 5, 7)
    plt.imshow(np.ones(C_img.shape))
    plt.title("B'")
    for c_idx, c in enumerate(last_r_images):
        c_img = cv2.imread(os.path.join(swig_images_path, c))[:,:,::-1]
        if c_idx <= 2:
            fig.add_subplot(2, 5, 2 + c_idx + 1)
        else:
            fig.add_subplot(2, 5, 5 + c_idx + 1)
        plt.imshow(c_img)
        plt.title(f"C{c_idx+1}")
    plt.tight_layout()
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    plt.grid(True)

    if out_p:
        plt.savefig(out_p)
        plt.close(fig)
        plt.cla()
    else:
        plt.show()

def analogy2str(analogy, verb, different_key):
    analogy['verb'] = verb
    analogy_len_keys_div_2 = int(len(analogy) / 2)
    analogy_no_diff_key = {k:v for k,v in analogy.items() if k != different_key}
    first_half = {k: v for i, (k, v) in enumerate(analogy_no_diff_key.items()) if i <= analogy_len_keys_div_2 and k != different_key}
    second_half = {k: v for i, (k, v) in enumerate(analogy_no_diff_key.items()) if i > analogy_len_keys_div_2 and k != different_key}
    diff_key_d = {k: v for i, (k, v) in enumerate(analogy.items()) if k == different_key}
    first_half_str = str(first_half).replace("{", "").replace("}", "")
    second_half_str = str(second_half).replace("{", "").replace("}", "")
    diff_key_str = str(diff_key_d).replace("{", "").replace("}", "")
    analogy_str = first_half_str
    if len(second_half) > 0:
        analogy_str += "\n" + second_half_str
    analogy_str += "\n" + r"$\bf{" + str(diff_key_str) + "}$"
    return analogy_str

def analogy2str_detailed(analogy, verb, different_key):
    analogy['verb'] = verb
    analogy_len_keys_div_3 = int(len(analogy) / 3)
    analogy_no_diff_key = {k:v for k,v in analogy.items() if k != different_key}
    first_third = {k: v for i, (k, v) in enumerate(analogy_no_diff_key.items()) if i <= analogy_len_keys_div_3 and k != different_key}
    second_third = {k: v for i, (k, v) in enumerate(analogy_no_diff_key.items()) if analogy_len_keys_div_3 < i <= analogy_len_keys_div_3 * 2 and k != different_key}
    third_third = {k: v for i, (k, v) in enumerate(analogy_no_diff_key.items()) if i > analogy_len_keys_div_3 * 2 and k != different_key}
    diff_key_d = {k: v for i, (k, v) in enumerate(analogy.items()) if k == different_key}
    first_third_str = str(first_third).replace("{", "").replace("}", "")
    second_third_str = str(second_third).replace("{", "").replace("}", "")
    third_third_str = str(third_third).replace("{", "").replace("}", "")
    diff_key_str = str(diff_key_d).replace("{", "").replace("}", "")
    analogy_str = first_third_str
    if len(second_third) > 0:
        analogy_str += "\n" + second_third_str
    if len(third_third) > 0:
        analogy_str += "\n" + third_third_str
    analogy_str += "\n" + r"$\bf{" + str(diff_key_str) + "}$"
    return analogy_str

def plot_distractors(img_path, annotations, verb, diff_key, relevant_candidates_sorted):
    relevant_candidates_sorted = relevant_candidates_sorted[:2]
    img = cv2.resize(cv2.imread(os.path.join(swig_images_path, img_path))[:, :, ::-1], (512, 512),
                     interpolation=cv2.INTER_AREA)
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(30, 10))
    axs[0].imshow(img)
    DIST_FONT_SIZE = 20
    axs[0].set_title(analogy2str_detailed(annotations, verb, diff_key),
                     fontsize=DIST_FONT_SIZE)
    for c_idx, c in enumerate(relevant_candidates_sorted):
        c_img = cv2.resize(cv2.imread(os.path.join(swig_images_path, c['img_name']))[:, :, ::-1], (512, 512),
                         interpolation=cv2.INTER_AREA)
        c_annotations = {k:v for k,v in c.items() if k not in ['img_name', 'verb']}
        c_verb = c['verb']
        c_annotations['sim_inp'] = c_annotations['cand_sim_with_input']
        # c_annotations['sim_sol'] = c_annotations['cand_sim_with_sol']
        del c_annotations['cand_sim_with_sol']
        del c_annotations['cand_sim_with_input']
        axs[c_idx + 1].imshow(c_img)
        axs[c_idx + 1].set_title(analogy2str_detailed(c_annotations, c_verb, diff_key),
                         fontsize=DIST_FONT_SIZE)
    plt.show()
