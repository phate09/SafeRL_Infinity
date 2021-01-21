from typing import List
import plotly.graph_objects as go
import numpy as np
import pypoman


def PolygonSort(corners):
    # e.g corners = [(0, 0), (3, 0), (2, 10), (3, 4), (1, 5.5)]
    n = len(corners)
    cx = float(sum(x for x, y in corners)) / n
    cy = float(sum(y for x, y in corners)) / n
    cornersWithAngles = []
    for x, y in corners:
        an = (np.arctan2(y - cy, x - cx) + 2.0 * np.pi) % (2.0 * np.pi)
        cornersWithAngles.append((x, y, an))
    cornersWithAngles.sort(key=lambda tup: tup[2])
    return cornersWithAngles  # map(lambda x: (x[0], x[1]), cornersWithAngles)


def compute_trace_polygons(polygons: List[List]):
    # e.g corners = [(0, 0), (3, 0), (2, 10), (3, 4), (1, 5.5)]
    # corners need to be sorted
    x_list = []
    y_list = []
    for corners in polygons:
        x = [corner[0] for corner in corners]
        y = [corner[1] for corner in corners]
        x.append(corners[0][0])
        y.append(corners[0][1])
        x.append(None)
        y.append(None)
        x_list.extend(x)
        y_list.extend(y)

    trace1 = go.Scatter(x=x_list, y=y_list, mode='markers', fill='toself', )
    return trace1


def transform_vertices(polygon_vertices_list):
    result = []
    for vertices in polygon_vertices_list:
        transformed_vertices = []
        for vertex in vertices:
            transformed_vertex = np.zeros(shape=(2,))
            transformed_vertex[0] = vertex[0] - vertex[1]
            transformed_vertex[1] = vertex[3]
            transformed_vertices.append(transformed_vertex)
        result.append(transformed_vertices)
    return result


def transform_vertices2(polygon_vertices_list):
    result = []
    for vertices in polygon_vertices_list:
        if vertices is not None:
            transformed_vertices = []
            for vertex in vertices:
                transformed_vertex = np.zeros(shape=(2,))
                transformed_vertex[0] = vertex[0]
                transformed_vertex[1] = vertex[1]
                transformed_vertices.append(transformed_vertex)
            result.append(transformed_vertices)
    return result


def show_polygon_list(polygon_vertices_list):  # x_prime_vertices, x_vertices

    # scaler = StandardScaler()
    # scaler.fit(polygon_vertices_list[0])  # list(itertools.chain.from_iterable(polygon_vertices_list)))
    # scaled_list = []
    # for x in polygon_vertices_list:
    #     scaled_list.append(scaler.transform(x))
    # pca = PCA(n_components=2)
    # pca.fit(list(itertools.chain.from_iterable(scaled_list)))  #
    # principal_components_list = []
    # for x_scaled in scaled_list:
    #     principal_components_list.append(pca.transform(x_scaled))
    traces = []
    for timestep in polygon_vertices_list:
        principal_components_list = transform_vertices(timestep)
        traces.append(compute_polygon_trace(principal_components_list))
    fig = go.Figure()
    for trace in traces:
        fig.add_trace(trace)
    fig.update_layout(xaxis_title="x_lead - x_ego", yaxis_title="Speed")
    fig.show()


def show_polygon_list2(polygon_vertices_list, y_axis_title="x_ego", x_axis_title="x_lead"):
    traces = []
    for timestep in polygon_vertices_list:
        principal_components_list = transform_vertices2(polygon_vertices_list[timestep])
        traces.append(compute_polygon_trace(principal_components_list))
    fig = go.Figure()
    for trace in traces:
        fig.add_trace(trace)
    fig.update_layout(xaxis_title=x_axis_title, yaxis_title=y_axis_title)
    fig.show()


def show_polygon_list3(polygon_vertices_list, x_axis_title, y_axis_title, template, template2d):
    traces = []
    projected_points = []
    for timestep in polygon_vertices_list:
        principal_components_list = transform_vertices2([windowed_projection(template, x, template2d)
                                                         for x in polygon_vertices_list[timestep]])
        traces.append(compute_polygon_trace(principal_components_list))
        projected_points.append([tuple([(y[0], y[1]) for y in PolygonSort(x)]) for x in principal_components_list])
    fig = go.Figure()
    for trace in traces:
        fig.add_trace(trace)
    fig.update_layout(xaxis_title=x_axis_title, yaxis_title=y_axis_title)
    return fig, projected_points


def create_window_boundary(template_input, x_results, template_2d, window_boundaries):
    assert len(window_boundaries) == 4
    window_template = np.vstack([template_input, template_2d, -template_2d])  # max and min
    window_boundaries = np.stack((window_boundaries[0], window_boundaries[1], -window_boundaries[2], -window_boundaries[3]))
    window_boundaries = np.concatenate((x_results, window_boundaries))
    # windowed_projection_max = np.maximum(window_boundaries, projection)
    # windowed_projection_min = np.minimum(window_boundaries, projection)
    # windowed_projection = np.concatenate((windowed_projection_max.squeeze(), windowed_projection_min.squeeze()), axis=0)
    return window_template, window_boundaries


def windowed_projection(template, x_results, template_2d):
    ub_lb_window_boundaries = np.array([1000, 1000, -1000, -1000])
    window_A, window_b = create_window_boundary(template, x_results, template_2d, ub_lb_window_boundaries)
    try:
        vertices, rays = pypoman.projection.project_polyhedron((template_2d, np.array([0, 0])), (window_A, window_b), canonicalize=False)
        if len(vertices) == 0:
            return None
    except:
        return None
    vertices = np.vstack(vertices)
    return vertices


def compute_polygon_trace(principalComponents: List[List]):
    polygon1 = [PolygonSort(x) for x in principalComponents]
    trace1 = compute_trace_polygons(polygon1)
    return trace1
